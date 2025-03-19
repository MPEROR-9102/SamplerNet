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
    def __init__(self, n_channels, n_classes, k_percent=0.4, bilinear=False):
        super(InterpolatorNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.k_percent = k_percent
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


def uni_sampling(targets, mask, k_percent=0.4):
    # Step 1: Flatten the tensors
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)
    
    # Step 2: Get indices where mask is 1
    valid_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    
    # Step 3: Determine the number of entries to sample
    num_samples = int(k_percent * valid_indices.numel())
    
    # Step 4: Perform uniform sampling
    # Select indices uniformly spaced across the valid indices
    step = max(1, valid_indices.numel() // num_samples)
    sampled_indices = valid_indices[torch.arange(0, valid_indices.numel(), step)[:num_samples]]
    
    # Step 5: Create a new tensor with all zeros
    output_flat = torch.zeros_like(targets_flat)
    
    # Step 6: Set sampled indices to values from the target tensor
    output_flat[sampled_indices] = targets_flat[sampled_indices]
    
    # Step 7: Reshape back to the original shape
    output = output_flat.view_as(targets)
    
    return output
    

def train(interpolator, train_hyper, train_loader, test_loader, loss_fn,\
          optimizer, scheduler, epochs, sample_type="random", save_path=""):
    
    best_val_loss, best_val_file = float("inf"), None
    if sample_type == "random":
        sample_func = rand_sampling
    else:
        sample_func = uni_sampling
        
    for epoch in range(epochs):
        interpolator.train()
        epoch_total_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            mask = sample_func(targets, (inputs[:,0] == 0).int().unsqueeze(1), interpolator.k_percent)
            interpolated_outputs = interpolator(torch.cat([inputs[:,0].unsqueeze(1), mask], dim=1))

            optimizer.zero_grad()
            loss = loss_fn(interpolated_outputs, targets)

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            
        # Epoch Summary
        num_batches = len(train_loader)        
        # Validation and LR scheduling
        val_total_loss = validate(interpolator, test_loader, sample_func, loss_fn)
        scheduler.step(epoch_total_loss/num_batches)
        
        if val_total_loss < best_val_loss:
            print(f"Epoch {epoch+1} Summary:")
            print(f"Avg Loss: {epoch_total_loss/num_batches:.6f}")
            print(f"Epoch {epoch+1} - Test Total Loss: {val_total_loss}")
            # model_dict = {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'test_loss': val_similarity_loss}
            ###
            # file_name = f"SamplerUNet_USC{train_hyper}_{epoch+1}_{model.target_sample_rate}_{val_similarity_loss}_pth.tar"
            # torch.save(model_dict, save_path + "/" + file_name) 
            best_val_loss = val_total_loss
            # best_val_file = file_name
            ###
            
    return best_val_loss, best_val_file


def validate(interpolator, test_loader, sample_func, loss_fn):
    interpolator.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            mask = sample_func(targets, (inputs[:,0] == 0).int().unsqueeze(1), interpolator.k_percent)
            interpolated_outputs = interpolator(torch.cat([inputs[:,0].unsqueeze(1), mask], dim=1))

            loss = loss_fn(interpolated_outputs, targets)
            total_loss += loss.item()
    
    return total_loss/len(test_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_hyper', type=float)
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
    
    best_test_losses, best_test_files = {}, {}
    start, end, interval = args.start_rate, args.end_rate, 2
    # path = "Final_USC_Save/"+f"{round(train_hyper, 1)}-{round(1 - train_hyper, 1)}-[{start}-{end}]"
    # if not os.path.exists(path):
    # 	os.mkdir(path)
    
    print("Training Started...")
    for target_sample_rate in np.linspace(start, end, interval):
        key = f"{round(train_hyper, 2)}-{round(1 - train_hyper, 2)}-{round(target_sample_rate, 2)}"    
        interpolator = InterpolatorNet(n_channels=2, n_classes=1, k_percent=target_sample_rate, bilinear=True).to(device)
        loss_fn = InterpolatorLoss().to(device)
        
        optimizer = optim.Adam(interpolator.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        best_test_loss, _ = train(interpolator, train_hyper, sampler_train, sampler_test, loss_fn,\
                                  optimizer, scheduler, sample_type='random', epochs=30)
        best_test_losses[key] = best_test_loss
        # best_test_files[key] = best_test_file
    
    print(best_test_losses)
    # print(best_test_files)
    print("Training Ends...")
