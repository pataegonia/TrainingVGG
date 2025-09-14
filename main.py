import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import os
import csv
import numpy as np

# 1. Dataset paths and hyperparameter settings
TRAIN_DATA_DIR = './train' # Assuming this now contains 'images' and 'masks' subdirectories
VAL_DATA_DIR = './val'     # Assuming this now contains 'images' and 'masks' subdirectories
BATCH_SIZE = 8  # U-Net often uses smaller batch sizes
LEARNING_RATE = 0.0001 # U-Net often uses lower learning rates with Adam
EPOCHS = 74
NUM_CLASSES = 2 # Binary segmentation (e.g., foreground/background). Adjust if multi-class.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './training_results_unet.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2. Data preprocessing and DataLoader setup (Modified for U-Net, assumes image and mask pairs)

class UNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png')) # Assuming masks are .png and have same base name
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Masks are typically grayscale

        if self.transform:
            # Apply same transformation to both image and mask
            # For segmentation, transforms must be carefully applied.
            # Here, we'll resize and then convert to tensor.
            # Normalization should only apply to the image.
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)
            
        return image, mask.long() # Masks should be long tensors for CrossEntropyLoss

# Define transforms for image and mask separately, then combine
image_transform = transforms.Compose([
    transforms.Resize((256, 256)), # U-Net paper used larger inputs (572x572), here for simplicity
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST), # Nearest for masks to preserve labels
    transforms.ToTensor(),
    # No normalization for masks
])

unet_transforms = {'image': image_transform, 'mask': mask_transform}

train_dataset = UNetDataset(root_dir=TRAIN_DATA_DIR, transform=unet_transforms)
val_dataset = UNetDataset(root_dir=VAL_DATA_DIR, transform=unet_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. U-Net Model Definition (according to U-Net paper)
class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
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

# 4. Initialize model, loss function, and optimizer (Modified for U-Net)
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE) # 3 channels for RGB images
criterion = nn.CrossEntropyLoss() # Standard for multi-class semantic segmentation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam is often preferred for U-Net

scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=10, verbose=True # Increased patience for U-Net
)

# Function to calculate Dice Score (common metric for segmentation)
def dice_score(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1) # Get predicted class for each pixel
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

# 5. Training and validation function (Modified for U-Net with Dice Score)
def train_model():
    best_dice_score = 0.0
    
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_dice_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, masks in train_loader:
                inputs, masks = inputs.to(DEVICE), masks.to(DEVICE).squeeze(1) # Squeeze mask channel if it's (B, 1, H, W) to (B, H, W)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            total_dice_score = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, masks in val_loader:
                    inputs, masks = inputs.to(DEVICE), masks.to(DEVICE).squeeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    total_dice_score += dice_score(outputs, masks).item()
            
            val_loss /= len(val_loader)
            avg_dice_score = total_dice_score / len(val_loader)

            scheduler.step(avg_dice_score) # Schedule based on Dice score

            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice Score: {avg_dice_score:.4f}")
            
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice_score': avg_dice_score
            })
            
            if avg_dice_score > best_dice_score:
                best_dice_score = avg_dice_score
                model_path = os.path.join(MODEL_SAVE_DIR, f'unet_best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved at: {model_path} with Dice Score: {best_dice_score:.4f}")

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("U-Net training and validation finished! Results saved to training_results_unet.csv.")