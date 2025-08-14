import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv

# 1. Dataset paths and hyperparameter settings
TRAIN_DATA_DIR = './train'
VAL_DATA_DIR = './val'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = './models'
RESULTS_FILE = './training_results.csv'

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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Custom CNN Model Definition
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 ,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Initialize model, loss function, and optimizer
model = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, mode='min', factor=0.1, patience=3)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

def set_dropout_p(m,p):
    if isinstance(m, nn.Dropout):
        m.p = p

def get_dropout_p(model):
    ps = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]
    return ps[0] if ps else 0.0

# 5. Training and validation function
def train_model():
    best_accuracy = 0.0
    
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(EPOCHS):
            # Dropout warmup
            if epoch < 30:
                model.apply(lambda m: set_dropout_p(m,0.0))
            else:
                model.apply(lambda m: set_dropout_p(m,0.3))
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
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)

            model.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    preds = model(x).argmax(1)
                    t_total += y.size(0)
                    t_correct += (preds == y).sum().item()
            train_acc = 100.0 * t_correct / max(1, t_total)

            if epoch in [30, 60]:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
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

            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"LR={cur_lr:.3e}  p(dropout)={get_dropout_p(model):.2f} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%"
            )

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
                print(f"New best model saved at: {model_path} "
                      f"with accuracy: {best_accuracy:.2f}%")
                
            scheduler.step()

# 6. Start training
if __name__ == '__main__':
    train_model()
    print("Training and validation finished! Results saved to training_results.csv.")
