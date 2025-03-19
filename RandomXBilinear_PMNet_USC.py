from __future__ import absolute_import, print_function
from __future__ import print_function, division
import subprocess
import sys

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
from collections import OrderedDict
from datetime import datetime
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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


def rand_sampling(targets, binary_tensor, k_percent):
    sampled_tensors = []
    for i in range(binary_tensor.size(0)):  # Iterate through batch
        tensor = binary_tensor[i, 0]  # Extract 256x256 tensor
        ones_indices = torch.nonzero(tensor, as_tuple=False)  # Indices of 1s
        k = int(k_percent * len(ones_indices))  # Calculate k%
        
        sampled_indices = ones_indices[torch.randperm(len(ones_indices))[:k]]  # Randomly sample k indices
        sampled_tensor = torch.zeros_like(tensor)
        sampled_tensor[sampled_indices[:, 0], sampled_indices[:, 1]] = 1  # Assign sampled indices to 1
        
        sampled_tensors.append(sampled_tensor.unsqueeze(0))
    
    return targets * torch.stack(sampled_tensors, dim=0)


def improved_interpolate_samplings(tensor, mode='bilinear', smooth=True, scale_factor=2):
    
    # Create a mask for valid (non-NaN) values
    mask = ~torch.isnan(tensor)
    # Replace NaNs with zeros for stable computation
    tensor_filled = torch.where(mask, tensor, torch.zeros_like(tensor))

    # Ensure the tensor is 4D (add a channel dimension if 3D)
    if tensor_filled.dim() == 3:
        tensor_filled = tensor_filled.unsqueeze(1)

    # Optionally smooth the tensor using Gaussian blur
    if smooth:
        tensor_filled = TVF.gaussian_blur(tensor_filled, kernel_size=(5, 5), sigma=(1.0, 1.0))

    # Upsample the tensor by the given scale factor
    upsampled = F.interpolate(tensor_filled, scale_factor=scale_factor, mode=mode, align_corners=False)
    # Downsample back to the original size
    interpolated_tensor = F.interpolate(upsampled, size=tensor.shape[-2:], mode=mode, align_corners=False)
    # Remove the added channel dimension
    interpolated_tensor = interpolated_tensor.squeeze(1)
    # Replace NaN regions in the original tensor with interpolated values
    result = torch.where(mask, tensor, interpolated_tensor)

    return result


def train(model, train_hyper, sample_type, sample_rate, train_loaders, test_loader, optimizer, scheduler, epochs, save_path=""):
    best_val_loss = float("inf")
    partial_samples_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_batches = sum(len(loader) for _, loader in train_loaders)

        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch") as pbar:
            for _type, train_loader in train_loaders:
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    if _type == 'partial':
                        with torch.no_grad():
                            targets = rand_sampling(targets, (inputs[:,0] == 0).int().unsqueeze(1), sample_rate).squeeze(1)
                            partial_samples_count += (targets > 0).int().sum()
                            targets = improved_interpolate_samplings(targets).unsqueeze(1)
                                    
                    preds = model(inputs)

                    optimizer.zero_grad()
                    loss = MSE(preds, targets)

                    loss.backward()
                    train_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    pbar.set_postfix(loss=f"{loss.item():.6f}")
                    pbar.update(1)

        scheduler.step()
        val_loss = validate(model, test_loader)

        if val_loss < best_val_loss:
            print(f'Epoch:{epoch+1}, Train Loss:{train_loss/total_batches:.6f}')
            print(f'Validating - Epoch:{epoch+1}, Test Loss:{val_loss:.6f}')
            
            # model_dict = {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'test_loss': val_loss}
            # file_name = f"PMNet_USC_{sample_type.capitalize()}_{train_hyper}_{epoch+1}_{sample_rate}_{val_loss:.6f}_pth.tar"
            # torch.save(model_dict, save_path + "/" + file_name)
            best_val_loss = val_loss

    return best_val_loss, partial_samples_count // epochs

def compensated_train(model, sample_rate, partials, train_loaders, test_loader, optimizer, scheduler, epochs):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_batches = sum(len(loader) for _, loader in train_loaders)
        partial_samples_count = partials

        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch") as pbar:
            for _type, train_loader in train_loaders:
                for inputs, targets in train_loader:
                    
                    if _type == "partial":
                        non_buildings_count = (targets > 0).int().sum()
                        if partial_samples_count < non_buildings_count:
                            break
                        else:
                            partial_samples_count -= non_buildings_count
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                                    
                    preds = model(inputs)

                    optimizer.zero_grad()
                    loss = MSE(preds, targets)

                    loss.backward()
                    train_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    pbar.set_postfix(loss=f"{loss.item():.6f}")
                    pbar.update(1)

        scheduler.step()
        val_loss = validate(model, test_loader)

        if val_loss < best_val_loss:
            print(f'Epoch:{epoch+1}, Train Loss:{train_loss/total_batches:.6f}')
            print(f'Validating - Epoch:{epoch+1}, Test Loss:{val_loss:.6f}')
            best_val_loss = val_loss

    return best_val_loss

def validate(model, test_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Validating", unit="batch") as pbar:
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                preds = model(inputs)
                preds = torch.clip(preds, 0, 1)

                loss = MSE(preds, targets)
                total_loss += loss.item()

                pbar.set_postfix(loss=f"{loss.item():.6f}")
                pbar.update(1)

        total_loss = total_loss / (len(test_loader) + 1e-7)

    return total_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_hyper', type=float)
    parser.add_argument('-s', '--sample_rate', type=float)
    args = parser.parse_args()
    
    data_root = "USC/"
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
    pmnet_test = DataLoader(pmnet_test_dataset, batch_size=16, shuffle=False, generator=torch.Generator(device="cpu"))

    lr = 5 * 1e-4
    lr_decay = 0.5
    step = 10
    sample_type = "random"
    target_sample_rate = args.sample_rate
    # save_path = "Final_USC_Save/PMNet_Unirand/"+f"{round(train_hyper, 1)}-{round(1 - train_hyper, 1)}-[{start}-{end}]"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    best_test_losses = {}

    
    #1 Fully-Sampled + Partially Sampled
    train_hyper = args.train_hyper
    train_full_size = int(pmnet_train_size * train_hyper)
    train_partial_size = pmnet_train_size - train_full_size
    train_full_dataset, train_partial_dataset = random_split(pmnet_train_dataset, [train_full_size, train_partial_size],\
                                           generator=torch.Generator(device="cpu"))
    train_full =  DataLoader(train_full_dataset, batch_size=16, shuffle=True, generator=torch.Generator(device="cpu"))
    train_partial = DataLoader(train_partial_dataset, batch_size=16, shuffle=True, generator=torch.Generator(device="cpu"))
    pmnet_train = [('full', train_full), ('partial', train_partial)]
    
    key = f"TH{round(train_hyper, 1)}-SR{round(target_sample_rate, 1)}"    
    model = PMNet(n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=lr_decay)
    print(key, "Training Started...")
    best_test_loss, partial_samples_count = train(model, train_hyper, sample_type, target_sample_rate, pmnet_train,\
                                           pmnet_test, optimizer, scheduler, epochs=30)
    best_test_losses[key] = best_test_loss
    print(key, "Partial Samples Count", partial_samples_count.item())

    
    #2 Compensated Fully-Sampled
    key = f"TH1.0-SR{round(target_sample_rate, 1)}"    
    model = PMNet(n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=lr_decay)
    print(key, "Training Started...")
    best_test_loss = compensated_train(model, target_sample_rate, partial_samples_count, pmnet_train, pmnet_test,\
                                       optimizer, scheduler, epochs=30)
    best_test_losses[key] = best_test_loss

    
    print(best_test_losses)
    print("Training Ends...")
    