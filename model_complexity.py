import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
import random

# SE-ResNet 모델들을 import하기 위해 필요한 클래스들을 정의
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.se = SEBlock(self.expansion * out_channels, reduction)

        self.shortcut = nn.Sequential()
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
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        return out

class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SEResNet, self).__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        
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

def SEResNet50(num_classes=10):
    return SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes)

def SEResNet101(num_classes=10):
    return SEResNet(SEBottleneck, [3, 4, 23, 3], num_classes)

def SEResNet152(num_classes=10):
    return SEResNet(SEBottleneck, [3, 8, 36, 3], num_classes)

def count_parameters(model):
    """모델의 총 파라미터 수를 계산"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """학습 가능한 파라미터 수를 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """FLOPs 계산 (근사치)"""
    def flop_count(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = output_elements * (kernel_size * input_channels + bias)
            output_dims = output.shape
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            groups = module.groups
            
            filters_per_channel = out_channels = module.out_channels
            conv_per_position_flops = int(torch.prod(torch.tensor(kernel_dims))) * in_channels // groups
            
            active_elements_count = int(torch.prod(torch.tensor(output_dims)))
            overall_conv_flops = conv_per_position_flops * active_elements_count
            
            # bias flops
            bias_flops = 0
            if module.bias is not None:
                bias_flops = out_channels * active_elements_count
            
            overall_flops = overall_conv_flops + bias_flops
            module.__flops__ += overall_flops
            
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = input_features * output_features + bias
            input_last_dim = input[0].shape[-1]
            bias_flops = module.out_features if module.bias is not None else 0
            module.__flops__ += input_last_dim * module.out_features + bias_flops

    # FLOPs 계산을 위한 hook 등록
    flop_counts = []
    
    def add_flops_counter_variable_or_reset(module):
        if hasattr(module, '__flops__'):
            module.__flops__ = 0
        else:
            module.__flops__ = 0

    def add_flops_counter_hook_function(module):
        if hasattr(module, '__flops_handle__'):
            return
        handle = module.register_forward_hook(flop_count)
        module.__flops_handle__ = handle

    def remove_flops_counter_hook_function(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__

    model.apply(add_flops_counter_variable_or_reset)
    model.apply(add_flops_counter_hook_function)

    # 더미 입력으로 forward pass
    with torch.no_grad():
        model.eval()
        _ = model(torch.randn(input_size))

    # 총 FLOPs 계산
    total_flops = 0
    for module in model.modules():
        if hasattr(module, '__flops__'):
            total_flops += module.__flops__

    # Hook 제거
    model.apply(remove_flops_counter_hook_function)
    
    return total_flops

def format_number(num):
    """숫자를 읽기 쉬운 형태로 포맷"""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def analyze_model_complexity():
    """각 SE-ResNet 모델의 복잡도 분석"""
    models = {
        'SE-ResNet-50': SEResNet50(),
        'SE-ResNet-101': SEResNet101(),
        'SE-ResNet-152': SEResNet152()
    }
    
    print("=" * 80)
    print("SE-ResNet Models Complexity Analysis")
    print("=" * 80)
    print(f"{'Model':<15} {'Parameters':<15} {'Trainable':<15} {'FLOPs':<15} {'Model Size (MB)':<15}")
    print("-" * 80)
    
    for name, model in models.items():
        # 파라미터 수 계산
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        
        # FLOPs 계산 (시간이 걸릴 수 있음)
        try:
            flops = calculate_flops(model)
        except:
            flops = 0  # FLOPs 계산 실패 시
        
        # 모델 크기 계산 (MB)
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 기준
        
        print(f"{name:<15} {format_number(total_params):<15} {format_number(trainable_params):<15} {format_number(flops):<15} {model_size_mb:.2f}MB")
    
    print("=" * 80)
    print("Note:")
    print("- Parameters: Total number of model parameters")
    print("- Trainable: Number of trainable parameters")
    print("- FLOPs: Floating Point Operations (approximate)")
    print("- Model Size: Memory size assuming float32 precision")
    print("=" * 80)

def detailed_layer_analysis(model, model_name):
    """레이어별 상세 분석"""
    print(f"\n{model_name} - Detailed Layer Analysis")
    print("-" * 60)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                print(f"{name:<40} {format_number(num_params):>15}")
                total_params += num_params
    
    print("-" * 60)
    print(f"{'Total':<40} {format_number(total_params):>15}")

if __name__ == '__main__':
    # 전체 복잡도 분석
    analyze_model_complexity()
    
    # 각 모델의 상세 분석 (선택사항)
    print("\n" + "="*80)
    print("Detailed Analysis (optional - uncomment to see layer-by-layer breakdown)")
    print("="*80)
    
    # 상세 분석을 원하면 아래 주석을 해제하세요
    # models = {
    #     'SE-ResNet-50': SEResNet50(),
    #     'SE-ResNet-101': SEResNet101(),
    #     'SE-ResNet-152': SEResNet152()
    # }
    # 
    # for name, model in models.items():
    #     detailed_layer_analysis(model, name)