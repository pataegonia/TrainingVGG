# model_analysis.py
import os
import csv
import torch
import torch.nn as nn
from torchinfo import summary  # pip install torchinfo

# ---------------------------
# MobileNet-like model 정의 (사용자 코드 그대로)
# ---------------------------
def make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x); x = self.bn_dw(x); x = self.relu(x)
        x = self.pointwise(x); x = self.bn_pw(x); x = self.relu(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, alpha: float = 1.0):
        super().__init__()
        c = lambda x: make_divisible(x * alpha)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConv(c(32),   c(64),   1),
            DepthwiseSeparableConv(c(64),   c(128),  2),
            DepthwiseSeparableConv(c(128),  c(128),  1),
            DepthwiseSeparableConv(c(128),  c(256),  2),
            DepthwiseSeparableConv(c(256),  c(256),  1),
            DepthwiseSeparableConv(c(256),  c(512),  2),
            DepthwiseSeparableConv(c(512),  c(512),  1),
            DepthwiseSeparableConv(c(512),  c(512),  1),
            DepthwiseSeparableConv(c(512),  c(512),  1),
            DepthwiseSeparableConv(c(512),  c(512),  1),
            DepthwiseSeparableConv(c(512),  c(512),  1),
            DepthwiseSeparableConv(c(512),  c(1024), 2),
            DepthwiseSeparableConv(c(1024), c(1024), 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

# ---------------------------
# 분석 유틸
# ---------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_model(image_size: int, alpha: float, num_classes: int = 10, device=DEVICE):
    """
    주어진 image_size, alpha에 대해 파라미터 수와 MACs를 반환합니다.
    """
    model = CustomCNN(num_classes=num_classes, alpha=alpha).to(device)
    model.eval()

    # torchinfo summary로 파라미터 및 Mult-Adds 계산
    info = summary(
        model,
        input_size=(1, 3, image_size, image_size),
        verbose=0,
        device=device
    )

    total_params = float(info.total_params)                # 개수
    # torchinfo 버전에 따라 total_mult_adds가 없을 수 있어 안전 처리
    total_macs = getattr(info, "total_mult_adds", None)    # 총 Mult-Adds (연산 수)

    return total_params, total_macs

def humanize(n, unit=""):
    if n is None:
        return "N/A"
    # 1e6 단위로 M(메가) 표기
    return f"{n/1_000_000:.2f} M{unit}"

# ---------------------------
# 메인: 조합 반복 분석 + CSV 저장
# ---------------------------
if __name__ == "__main__":
    # 원하는 조합(예시): 이미지 크기와 alpha
    IMAGE_SIZES = [128, 160, 192, 224]  # stride=2가 여러 번 있으므로 32의 배수/근처 권장
    ALPHAS = [0.25, 0.5, 0.75, 1.0]

    RESULTS_FILE = "./model_analysis_results.csv"
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)

    rows = []
    print("============== Model Analysis (Params & MACs) ==============")
    print(f"{'Image':>6} | {'alpha':>5} | {'Params (M)':>11} | {'MACs (M Mult-Adds)':>21}")
    print("-" * 54)

    for sz in IMAGE_SIZES:
        for a in ALPHAS:
            params, macs = analyze_model(sz, a)
            print(f"{sz:>6} | {a:>5} | {humanize(params):>11} | {humanize(macs, ' Mult-Adds'):>21}")
            rows.append({
                "image_size": sz,
                "alpha": a,
                "total_params": params,
                "total_params_M": params/1_000_000,
                "total_mult_adds": macs if macs is not None else "",
                "total_mult_adds_M": (macs/1_000_000) if macs is not None else ""
            })

    print("============================================================")

    # CSV 저장
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_size", "alpha", "total_params", "total_params_M",
                        "total_mult_adds", "total_mult_adds_M"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"결과가 CSV로 저장되었습니다: {RESULTS_FILE}")
