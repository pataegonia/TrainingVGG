import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import csv

# 1. Dataset paths and hyperparameter settings from the ResNet paper
TRAIN_DATA_DIR = './train'
VAL_DATA_DIR = './val'
# The paper uses a mini-batch size of 256 for ImageNet training.
BATCH_SIZE = 256
# The paper trained for up to 600,000 iterations. We'll set a fixed number of epochs.
EPOCHS = 90
LEARNING_RATE = 0.1
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './training_results.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class RandomShortSideResize:
    """
    As described in the ResNet paper for scale augmentation, the shorter side of the image
    is randomly sampled in [256, 480].
    """
    def __init__(self, min_size=256, max_size=480):
        self.min_size = min_size
        self.max_size = max_size
    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        w, h = img.size
        if w < h:
            new_w, new_h = size, int(h * size / w)
        else:
            new_h, new_w = size, int(w * size / h)
        return TF.resize(img, (new_h, new_w))

# 2. Data preprocessing and DataLoader setup based on the ResNet paper
train_transform = transforms.Compose([
    # Scale augmentation as per the paper
    RandomShortSideResize(256, 480),
    # Random 224x224 crop and horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224),
    # Color augmentation as mentioned in the paper
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    ),
    transforms.ToTensor(),
    # Normalization with ImageNet mean and std
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# For validation, a single-scale evaluation is used as per the paper
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 3. ResNet-34 Model Definition
# This class is structured to follow the ResNet architecture described in the paper.
class BasicBlock(nn.Module):
    """
    The basic residual block for ResNet-18 and ResNet-34.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # The shortcut connection needs to downsample the input if the stride is not 1
        # or if the number of input and output channels are different.
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution layer (stem) as described in the paper
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # The four main layers of the ResNet architecture
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # The final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization as per the paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# 4. Initialize model, loss function, and optimizer as per the paper
model = ResNet34(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
# The paper uses SGD with momentum and weight decay
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True # Nesterov momentum is a common improvement
)

# The paper mentions dividing the learning rate by 10 when the validation error plateaus.
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10, # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=1e-6
)

# 5. Training and validation function
def train_model():
    best_accuracy = 0.0
    
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            train_loss = running_loss / len(train_dataset)
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_running_loss / len(val_dataset)
            accuracy = 100 * correct / total
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, LR: {current_lr}")
            
            scheduler.step(val_loss)
            
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': accuracy,
                'learning_rate': current_lr
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_resnet34.pth')
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved at: {model_path} with accuracy: {best_accuracy:.2f}%")

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("Training and validation finished! Results saved to training_results.csv.")