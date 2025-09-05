import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
from torchinfo import summary  # pip install torchinfo

# 1. Dataset paths and hyperparameter settings
TRAIN_DATA_DIR = './train'
VAL_DATA_DIR = './val'
BATCH_SIZE = 96
LEARNING_RATE = 0.01
WEIGHT_DECAY = 4e-5
EPOCHS = 100
NUM_CLASSES = 10
IMAGE_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './0.5MobileNet128.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2. Data preprocessing and DataLoader setup
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # color distortions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Custom CNN Model Definition
# "Baseline" CNN, 논문의 CNN 구조를 참조하여 CustomCNN class 내부 구성을 수정하세요.

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)     # ✅ depthwise용 BN

        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)    # ✅ pointwise용 BN
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)    # ✅ in_channels 크기와 일치
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)    # ✅ out_channels 크기와 일치
        x = self.relu(x)
        return x
    

def make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)
    

class CustomCNN(nn.Module):
    def __init__(self, num_classes, alpha: float = 1.0):
        super(CustomCNN, self).__init__()
        c=lambda x: make_divisible(x * alpha)
        # Initial standard convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True)
        )

        # MobileNet body
        self.features = nn.Sequential(
            DepthwiseSeparableConv(c(32),  c(64),  1),
            DepthwiseSeparableConv(c(64),  c(128), 2),
            DepthwiseSeparableConv(c(128), c(128), 1),
            DepthwiseSeparableConv(c(128), c(256), 2),
            DepthwiseSeparableConv(c(256), c(256), 1),
            DepthwiseSeparableConv(c(256), c(512), 2),
            DepthwiseSeparableConv(c(512), c(512), 1),
            DepthwiseSeparableConv(c(512), c(512), 1),
            DepthwiseSeparableConv(c(512), c(512), 1),
            DepthwiseSeparableConv(c(512), c(512), 1),
            DepthwiseSeparableConv(c(512), c(512), 1),
            DepthwiseSeparableConv(c(512), c(1024), 2),
            DepthwiseSeparableConv(c(1024), c(1024), 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.001),
            nn.Linear(c(1024), num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Initialize model, loss function, and optimizer
model = CustomCNN(num_classes=NUM_CLASSES, alpha=0.5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    alpha=0.9, 
    eps=0.1,
    weight_decay=WEIGHT_DECAY
)
scheduler = ExponentialLR(optimizer, gamma=0.98)


print("================== Model Analysis ==================")
input_size = (1, 3, 224, 224)
summary_info = summary(model, input_size=input_size, verbose=0, device=DEVICE)

total_params = summary_info.total_params / 1_000_000
print(f"Total Parameters: {total_params:.2f} Million")

# MACs (Million Mult-Adds)
total_macs = summary_info.total_mult_adds / 1_000_000
print(f"Total MACs (Mult-Adds): {total_macs:.2f} Million")
print("====================================================")

# 5. Training and validation function
def train_model():
    best_accuracy = 0.0
    
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            running_loss = 0.0
            # Wrap the DataLoader with tqdm for a progress bar
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            # Wrap the DataLoader with tqdm for a progress bar
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
            
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(MODEL_SAVE_DIR, f'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved at: {model_path} with accuracy: {best_accuracy:.2f}%")

            scheduler.step()

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("Training and validation finished! Results saved to training_results.csv.")