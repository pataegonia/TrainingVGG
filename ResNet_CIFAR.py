import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import math
import csv

# 1. Dataset paths and hyperparameter settings
TRAIN_DATA_DIR = './train'
VAL_DATA_DIR = './val'
BATCH_SIZE = 128
TARGET_STEPS = 64000
LEARNING_RATE = 0.1
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './training_results_10.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2. Data preprocessing and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
EPOCHS = math.ceil(TARGET_STEPS / steps_per_epoch)

# 3. Custom CNN Model Definition
# "Baseline" CNN, 논문의 CNN 구조를 참조하여 CustomCNN class 내부 구성을 수정하세요.
N = 3

class CustomCNN(nn.Module):
    # CIFAR-10 설정: option A (identity shortcut + zero-pad), 3x3-3x3 BasicBlock
    class Residual(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.in_ch  = in_ch
            self.out_ch = out_ch
            self.stride = stride

            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            # --- Option A: identity shortcut ---
            if self.stride != 1:  # 해상도 줄일 때
                identity = F.avg_pool2d(identity, kernel_size=1, stride=self.stride)
            if self.in_ch != self.out_ch:  # 채널 늘릴 때 0-padding
                pad_c = self.out_ch - self.in_ch
                zeros = torch.zeros(identity.size(0), pad_c, identity.size(2), identity.size(3),
                                    device=identity.device, dtype=identity.dtype)
                identity = torch.cat([identity, zeros], dim=1)

            out = self.relu(out + identity)
            return out

    def __init__(self, num_classes, n: int = N):
        super(CustomCNN, self).__init__()

        # --- stem: 3x3 conv, C=16 (stride=1), no pooling ---
        stem = [
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]

        # stage 구성: 각 stage에 블록 n개(= 2n conv layers), 첫 블록만 stride=2
        # 출력 맵 크기: 32x32 -> 16x16 -> 8x8, 채널: 16 -> 32 -> 64
        l1 = [CustomCNN.Residual(16, 16, stride=1) for _ in range(n)]
        l2 = [CustomCNN.Residual(16, 32, stride=2)] + \
             [CustomCNN.Residual(32, 32, stride=1) for _ in range(n-1)]
        l3 = [CustomCNN.Residual(32, 64, stride=2)] + \
             [CustomCNN.Residual(64, 64, stride=1) for _ in range(n-1)]

        # features = stem + (stage1,2,3)
        self.features = nn.Sequential(*stem, *l1, *l2, *l3)

        # head: Global Average Pooling -> FC(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

        # He(Kaiming) 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)     # 32x32 -> 16x16 -> 8x8
        x = self.avgpool(x)      # 1x1
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Initialize model, loss function, and optimizer
model = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # val_loss 하락이 멈추면 plateau로 판단
    factor=0.1,        # LR = LR * 0.1
    patience=3,        # 3 epoch 동안 개선 없을 때 감쇠
    cooldown=0,        # 감쇠 후 바로 다시 모니터링
    min_lr=1e-6,       # 너무 작아지는 것 방지
    threshold=1e-4,    # 개선 판단 임계
    threshold_mode='rel'
)

def current_lr(optim_):
    # (SGD는 param group 하나일 가능성이 높음)
    return optim_.param_groups[0]['lr']

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

            scheduler.step(val_loss)
            
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

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("Training and validation finished! Results saved to training_results.csv.")