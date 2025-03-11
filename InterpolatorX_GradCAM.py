from __future__ import absolute_import, print_function
from collections import OrderedDict
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss as MSE

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4


seed = 42
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(seed)
print(f"Seeds set to {seed} for deterministic results.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Session Device: {device} & {torch.cuda.device_count()}")


class PMnet_usc(Dataset):
    def __init__(self, csv_file,
                 dir_dataset="USC/",               
                 transform= transforms.ToTensor()):
        
        self.ind_val = pd.read_csv(csv_file)
        self.dir_dataset = dir_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ind_val)
    
    def __getitem__(self, idx):

        #Load city map
        self.dir_buildings = self.dir_dataset+ "map/"
        img_name_buildings = os.path.join(self.dir_buildings, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        self.dir_Tx = self.dir_dataset+ "Tx/" 
        img_name_Tx = os.path.join(self.dir_Tx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Tx = np.asarray(io.imread(img_name_Tx))

        #Load Rx (reciever): (not used in our training)
        self.dir_Rx = self.dir_dataset+ "Rx/" 
        img_name_Rx = os.path.join(self.dir_Rx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Rx = np.asarray(io.imread(img_name_Rx))

        #Load Power:
        self.dir_power = self.dir_dataset+ "pmap/" 
        img_name_power = os.path.join(self.dir_power, str(self.ind_val.iloc[idx, 0])) + ".png"
        image_power = np.asarray(io.imread(img_name_power))        

        inputs=np.stack([image_buildings, image_Tx], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            power = self.transform(image_power).type(torch.float32)

        return [inputs , power]


# Conv, Batchnorm, Relu layers, basic building block.
class _ConvBnReLU(nn.Sequential):

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

# Bottleneck layer cinstructed from ConvBnRelu layer block, buiding block for Res layers
class _Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

# Res Layer used to costruct the encoder
class _ResLayer(nn.Sequential):

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

# Stem layer is the initial interfacing layer
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch, in_ch = 2):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(in_ch, 2, 1, ceil_mode=True))



class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


# Atrous spatial pyramid pooling
class _ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)



# Decoder layer constricted using these 2 blocks
def ConRu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

def ConRuT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )



class PMNet(nn.Module):

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride):
        super(PMNet, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)

        # Decoder
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRuT(512+512, 512, 3, 1)
        self.conv_up3 = ConRuT(512+512, 256, 3, 1)
        self.conv_up2 = ConRu(256+256, 256, 3, 1)
        self.conv_up1 = ConRu(256+256, 256, 3, 1)

        self.conv_up0 = ConRu(256+64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
                         nn.Conv2d(128+2, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        xup00 = self.conv_up00(xup0)
        
        return xup00


class MultiLayerGradCAM:
    """
    Enhanced GradCAM implementation for PMNet that combines visualizations from multiple layers
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.layer_activations = {}
        self.layer_gradients = {}
        self.hooks = []
    
    def _register_hooks(self, target_layers):
        """Register hooks to capture activations and gradients for multiple layers"""
        for layer_name, layer in target_layers.items():
            # Forward hook to capture activations
            forward_hook = layer.register_forward_hook(
                lambda module, input, output, name=layer_name: 
                    self.layer_activations.update({name: output.detach()})
            )
            
            # Backward hook to capture gradients
            backward_hook = layer.register_full_backward_hook(
                lambda module, grad_in, grad_out, name=layer_name: 
                    self.layer_gradients.update({name: grad_out[0].detach()})
            )
            
            self.hooks.extend([forward_hook, backward_hook])
    
    def _clean_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _get_layer_by_name(self, layer_name):
        """Helper function to get layer by name"""
        if layer_name == 'fc1':
            return self.model.fc1
        elif layer_name == 'aspp':
            return self.model.aspp
        elif layer_name.startswith('layer'):
            layer_num = int(layer_name[5:])
            return getattr(self.model, f'layer{layer_num}')
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
    
    def generate(self, input_tensor, layer_names=None, weights=None, target_mask=None):
        """
        Generate GradCAM heatmap from multiple layers
        
        Args:
            input_tensor: Input to model with shape [B, 2, 256, 256]
            layer_names: List of layer names to use for GradCAM 
                        Default: ['fc1', 'layer5', 'layer4']
            weights: Optional list of weights for each layer (must match length of layer_names)
                    Default: Equal weights
            target_mask: Optional mask to focus on specific regions
            
        Returns:
            combined_cam: Combined GradCAM heatmap with shape [B, 256, 256]
        """
        # Default layers if none specified
        if layer_names is None:
            layer_names = ['fc1', 'layer5', 'layer4']
        
        # Default weights if none specified
        if weights is None:
            weights = [1.0/len(layer_names)] * len(layer_names)
        elif len(weights) != len(layer_names):
            raise ValueError("Number of weights must match number of layers")
        
        # Reset stored activations and gradients
        self.layer_activations = {}
        self.layer_gradients = {}
        
        # Get target layers
        target_layers = {name: self._get_layer_by_name(name) for name in layer_names}
        
        # Register hooks for all target layers
        self._register_hooks(target_layers)
        
        # Ensure input is on same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Forward pass to get model output
        self.model.zero_grad()
        output = self.model(input_tensor)  # [B, 1, 256, 256]
        
        # Make sure target_mask is on the same device if provided
        if target_mask is not None:
            target_mask = target_mask.to(device)
            # Ensure mask has correct shape
            if target_mask.dim() == 3:  # [B, 256, 256]
                target_mask = target_mask.unsqueeze(1)  # [B, 1, 256, 256]
            # Focus on the region defined by the mask
            loss = (output * target_mask).sum()
        else:
            # Use the entire output
            loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check if we have captured activations and gradients
        if not self.layer_activations or not self.layer_gradients:
            self._clean_hooks()
            raise RuntimeError("Failed to capture activations or gradients")
        
        # Process each layer's GradCAM and store
        layer_cams = []
        for i, layer_name in enumerate(layer_names):
            weight = weights[i]
            
            # Skip if layer data wasn't captured
            if layer_name not in self.layer_activations or layer_name not in self.layer_gradients:
                print(f"Warning: Layer {layer_name} data wasn't captured properly. Skipping.")
                continue
                
            # Get activations and gradients for this layer
            activations = self.layer_activations[layer_name]
            gradients = self.layer_gradients[layer_name]
            
            # Global average pooling of gradients (along spatial dimensions)
            alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Weight the activation maps
            if len(activations.shape) == 4:  # Standard case for conv layers
                cam = torch.sum(alpha * activations, dim=1)
            else:
                # Handle special cases (like ASPP output which might be reshaped)
                print(f"Warning: Unusual activation shape for {layer_name}: {activations.shape}")
                continue
            
            # Apply ReLU to focus on positive influence
            cam = torch.relu(cam)
            
            # Normalize CAM (per sample)
            cam_normalized = torch.zeros_like(cam)
            for j in range(cam.shape[0]):
                if torch.max(cam[j]) > 0:
                    cam_normalized[j] = cam[j] / torch.max(cam[j])
            
            # Resize CAM to match input dimensions if necessary
            if cam_normalized.shape != input_tensor.shape[2:]:
                cam_normalized = torch.nn.functional.interpolate(
                    cam_normalized.unsqueeze(1),
                    size=input_tensor.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            # Add to the list with corresponding weight
            layer_cams.append(cam_normalized * weight)
        
        # Combine all layer CAMs
        combined_cam = torch.zeros_like(layer_cams[0])
        for cam in layer_cams:
            combined_cam += cam
            
        # Renormalize the combined CAM
        for j in range(combined_cam.shape[0]):
            if torch.max(combined_cam[j]) > 0:
                combined_cam[j] = combined_cam[j] / torch.max(combined_cam[j])
        
        # Clean up hooks
        self._clean_hooks()
        
        return combined_cam
    
    def visualize(self, input_tensor, cam, output=None, n_samples=4, save_path=None, layer_names=None):
        """
        Visualize GradCAM results
        
        Args:
            input_tensor: Original input to model [B, 2, 256, 256]
            cam: GradCAM heatmap [B, 256, 256]
            output: Optional model output [B, 1, 256, 256]
            n_samples: Number of samples to visualize
            save_path: Optional path to save figure
            layer_names: Layer names used for the visualization (for title)
        """
        # Convert to numpy - make sure to move tensors to CPU first
        input_np = input_tensor.detach().cpu().numpy()
        cam_np = cam.detach().cpu().numpy()
        
        # Create title for the visualization
        if layer_names:
            cam_title = f"Multi-Layer GradCAM ({', '.join(layer_names)})"
        else:
            cam_title = "Multi-Layer GradCAM"
        
        # Determine number of columns in the plot
        n_cols = 4 if output is not None else 3
        
        # Create figure
        fig, axes = plt.subplots(
            min(n_samples, input_np.shape[0]), 
            n_cols,
            figsize=(5*n_cols, 5*min(n_samples, input_np.shape[0]))
        )
        
        # Handle single sample case
        if n_samples == 1:
            axes = np.array([axes])
        
        for i in range(min(n_samples, input_np.shape[0])):
            # Plot first input channel
            axes[i, 0].imshow(input_np[i, 0], cmap='gray')
            axes[i, 0].set_title(f"Input Ch1 - Sample {i+1}")
            axes[i, 0].axis('off')
            
            # Plot second input channel
            axes[i, 1].imshow(input_np[i, 1], cmap='gray')
            axes[i, 1].set_title(f"Input Ch2 - Sample {i+1}")
            axes[i, 1].axis('off')
            
            # Plot GradCAM heatmap
            heatmap = axes[i, 2].imshow(cam_np[i], cmap='jet')
            axes[i, 2].set_title(f"{cam_title} - Sample {i+1}")
            axes[i, 2].axis('off')
            plt.colorbar(heatmap, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Plot model output if provided
            if output is not None:
                output_np = output.detach().cpu().numpy()
                axes[i, 3].imshow(output_np[i, 0], cmap='gray')
                axes[i, 3].set_title(f"Model Output - Sample {i+1}")
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class InterpolatorNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(InterpolatorNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, inputs):
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class InterpolatorLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(InterpolatorLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE

    def forward(self, y_pred, y_true):
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        return self.alpha * mse_loss


def top_k_masking(x, mask, k_percent=0.2):
    # Get masked elements
    masked_x = x * mask  # Mask out x values where mask is 0
    
    # Flatten last two dimensions for processing
    flattened_x = masked_x.view(x.size(0), x.size(1), -1)
    flattened_mask = mask.view(mask.size(0), mask.size(1), -1)
    
    # Compute number of elements to keep for each tensor
    masked_counts = flattened_mask.sum(dim=-1, keepdim=True)
    top_k_counts = (masked_counts * k_percent).to(torch.int64)  # Cast to int64
    
    # Sort masked elements in descending order along the last axis
    sorted_x, _ = torch.sort(flattened_x, dim=-1, descending=True)
    
    # Get threshold values for top k%
    top_k_thresholds = sorted_x.gather(
        dim=-1, index=top_k_counts.clamp(min=1) - 1
    )
    
    # Broadcast thresholds back to the original tensor shape
    thresholds = top_k_thresholds.unsqueeze(-1)
    
    # Set top k% values to 1 (where mask == 1)
    result = torch.where(
        (x > thresholds) & mask,  # Condition
        torch.tensor(1.0, device=x.device),  # True case
        torch.tensor(0.0, device=x.device)
    )
    
    return result, masked_counts


def train(gradcam, layer_names, interpolator, k_percent, train_loader,
          test_loader, loss_fn, optimizer, scheduler, epochs, save_path=""):
    best_val_loss = float("inf")
        
    for epoch in range(epochs):
        interpolator.train()
        epoch_total_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            cam = gradcam.generate(inputs, layer_names=layer_names).unsqueeze(1).detach()
            # print(cam.shape)
            with torch.no_grad():
                masked_output, _ = top_k_masking(cam, (inputs[:,0]==0).unsqueeze(1), k_percent)
                # print(masked_output.shape)
                masked_output = masked_output * targets
            # plt.imshow(masked_output[0, 0].detach().cpu().numpy())
            # plt.show()
            # sys.exit(0)
                
            outputs = interpolator(torch.cat([inputs[:,0].unsqueeze(1), masked_output], dim=1))

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            
        num_batches = len(train_loader)        
        val_total_loss = validate(gradcam, layer_names, interpolator, k_percent, test_loader, loss_fn)
        scheduler.step(epoch_total_loss/num_batches)
        
        
        if val_total_loss < best_val_loss:
            print(f"Epoch {epoch+1} Summary:")
            print(f"Avg Loss: {epoch_total_loss/num_batches:.6f}")
            print(f"Epoch {epoch+1} - Test Total Loss: {val_total_loss}")
#             # model_dict = {
#             #     'model_state_dict': model.state_dict(),
#             #     'optimizer_state_dict': optimizer.state_dict(),
#             #     'test_loss': val_similarity_loss}
#             ###
#             # file_name = f"SamplerUNet_USC{train_hyper}_{epoch+1}_{model.target_sample_rate}_{val_similarity_loss}_pth.tar"
#             # torch.save(model_dict, save_path + "/" + file_name) 
            best_val_loss = val_total_loss
#             ###
            
    return best_val_loss


def validate(gradcam, layer_names, interpolator, k_percent, test_loader, loss_fn):
    interpolator.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.set_grad_enabled(True):
                cam = gradcam.generate(inputs, layer_names=layer_names).unsqueeze(1).detach()
            masked_output, _ = top_k_masking(cam, (inputs[:,0]==0).unsqueeze(1), k_percent)
            masked_output = masked_output * targets
                
            interpolated_outputs = interpolator(torch.cat([inputs[:,0].unsqueeze(1), masked_output], dim=1))

            loss = loss_fn(interpolated_outputs, targets)
            total_loss += loss.item()
    
    return total_loss/len(test_loader)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_hyper', type=float)
    parser.add_argument('-l', '--layer_names', type=str)
    parser.add_argument('-s', '--start_rate', type=float)
    parser.add_argument('-e', '--end_rate', type=float)
    args = parser.parse_args()
    
    data_root = "/home1/mathiyar/USC/"
    csv_file = os.path.join(data_root,'Data_coarse_train.csv')
    num_of_maps = 19016
    ddf = pd.DataFrame(np.arange(1,num_of_maps))
    ddf.to_csv('usc.csv', index=False)
    
    pathloss_data = PMnet_usc(csv_file = "usc.csv", dir_dataset=data_root)
    dataset_size = len(pathloss_data)
    
    pmnet_train_size = int(dataset_size * 0.8)
    pmnet_test_size = dataset_size - pmnet_train_size
    pmnet_train_dataset, pmnet_test_dataset = random_split(pathloss_data, [pmnet_train_size, pmnet_test_size], \
                                            generator=torch.Generator(device="cpu"))
    
    train_hyper = args.train_hyper
    sampler_train_size = int(pmnet_train_size * train_hyper)
    sampler_test_size = pmnet_train_size - sampler_train_size
    sampler_train_dataset, sampler_test_dataset = random_split(pmnet_train_dataset, [sampler_train_size, sampler_test_size], \
                                            generator=torch.Generator(device="cpu"))
    sampler_train =  DataLoader(sampler_train_dataset, batch_size=16, shuffle=True, generator=torch.Generator(device="cpu"))
    sampler_test = DataLoader(sampler_test_dataset, batch_size=16, shuffle=False, generator=torch.Generator(device="cpu"))
    
    pmnet = PMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16)
    pmnet.to(device)
    pmnet.load_state_dict(torch.load('USC_16H_16W.pt'))
    
    multi_gradcam = MultiLayerGradCAM(pmnet)
    layer_names = args.layer_names.split("|")
    print(layer_names)
    
    best_test_losses = {}
    start, end, interval = args.start_rate, args.end_rate, 2
    
    print("Training Started...")
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for target_sample_rate in np.linspace(start, end, interval):
        key = f"TH{round(train_hyper, 1)}-{"|".join(layer_names)}-SR{round(target_sample_rate, 2)}"
        
        interpolator = InterpolatorNet(n_channels=2, n_classes=1, bilinear=True).to(device)
        optimizer = optim.Adam(interpolator.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        loss_fn = InterpolatorLoss().to(device)
        
        best_test_losses[key] = train(multi_gradcam, layer_names, interpolator, target_sample_rate, sampler_train, sampler_test,\
                                      loss_fn, optimizer, scheduler, epochs=30)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(best_test_losses)
    print("Training Ends...")
