import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import vgg11_bn
import os, csv

# 1. Dataset paths and hyperparameter settings
TRAIN_DATA_DIR = './train'
VAL_DATA_DIR = './val'
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 74
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './training_results_a.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2. Data preprocessing and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# 3. Custom CNN (VGG16-style + 일부 BN)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 ,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)   # ← 마지막 층은 데이터셋에 맞게 유지
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ----- (A) He/Kaiming 초기화 -----
def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ----- (B) VGG11(BN) → CustomCNN 사전초기화 -----
def preinit_from_vgg11_bn(target_model):
    # torchvision 버전에 따라 weights API/ pretrained API 호환
    try:
        from torchvision.models import VGG11_BN_Weights
        src = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    except Exception:
        src = vgg11_bn(pretrained=True)

    # 순서대로 Conv/BN/Linear만 뽑아서 shape가 같을 때만 복사
    def collect(m):
        return [mod for mod in m.modules()
                if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))]

    s_layers = collect(src)
    t_layers = collect(target_model)

    s_idx, copied = 0, 0
    for t in t_layers:
        for k in range(s_idx, len(s_layers)):
            s = s_layers[k]
            if type(s) is not type(t):
                continue
            ok = False
            if isinstance(t, nn.Conv2d):
                ok = (s.weight.shape == t.weight.shape)
            elif isinstance(t, nn.BatchNorm2d):
                ok = (s.weight.shape == t.weight.shape and s.bias.shape == t.bias.shape)
            else:  # Linear
                ok = (s.weight.shape == t.weight.shape and s.out_features == t.out_features)

            if ok:
                with torch.no_grad():
                    t.weight.copy_(s.weight)
                    if hasattr(t, 'bias') and t.bias is not None and s.bias is not None:
                        t.bias.copy_(s.bias)
                    if isinstance(t, nn.BatchNorm2d):
                        t.running_mean.copy_(s.running_mean)
                        t.running_var.copy_(s.running_var)
                        if hasattr(t, 'num_batches_tracked') and hasattr(s, 'num_batches_tracked'):
                            t.num_batches_tracked.copy_(s.num_batches_tracked)
                copied += 1
                s_idx = k + 1
                break
    print(f"[Preinit] Copied {copied} layers from VGG11_BN.")

# 4. Initialize model, loss function, and optimizer
model = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.apply(kaiming_init)             # 기본 초기화
preinit_from_vgg11_bn(model)          # 동일 모양 층만 사전초기화

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    [
        {'params': model.features.parameters(),   'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-2},  # 마지막 Linear 포함
    ],
    momentum=0.9, weight_decay=5e-4, nesterov=False
)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)


# 5. Training and validation
def train_model():
    best_accuracy = 0.0
    max_decays, decays_done = 3, 0
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss','val_accuracy'])
        writer.writeheader()

        for epoch in range(EPOCHS):
            # Train
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)

            # Validate
            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
            val_loss /= len(val_loader)
            accuracy = 100.0 * correct / total

            # LR scheduling (val acc 기준, 논문 방식)
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(accuracy)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr:
                decays_done += 1
                print(f"[LR] {prev_lr:.6f} -> {new_lr:.6f} (decays={decays_done})")
                if decays_done >= max_decays:
                    print("LR을 3회 줄여 학습을 종료합니다.")
                    # 결과 저장 후 종료
                    print(f"Epoch [{epoch+1}/{EPOCHS}] "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")
                    break

            print(f"Epoch [{epoch+1}/{EPOCHS}], "
                  f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                  f"Validation Accuracy: {accuracy:.2f}%")

            writer.writerow({'epoch': epoch+1, 'train_loss': train_loss,
                             'val_loss': val_loss, 'val_accuracy': accuracy})

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                path = os.path.join(MODEL_SAVE_DIR, 'best_model1.pth')
                torch.save(model.state_dict(), path)
                print(f"New best model saved at: {path} with accuracy: {best_accuracy:.2f}%")

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("Training and validation finished! Results saved to training_results.csv.")
