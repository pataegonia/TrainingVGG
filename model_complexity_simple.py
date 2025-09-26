"""
SE-ResNet 모델 복잡도 분석 (PyTorch 없이)
각 모델의 이론적 파라미터 수와 구조를 계산합니다.
"""
import csv
import os

def calculate_conv2d_params(in_channels, out_channels, kernel_size, bias=False):
    """Conv2D 레이어의 파라미터 수 계산"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    weight_params = in_channels * out_channels * kernel_size[0] * kernel_size[1]
    bias_params = out_channels if bias else 0
    return weight_params + bias_params

def calculate_batchnorm_params(num_features):
    """BatchNorm 레이어의 파라미터 수 계산"""
    return num_features * 2  # weight + bias

def calculate_linear_params(in_features, out_features, bias=True):
    """Linear 레이어의 파라미터 수 계산"""
    weight_params = in_features * out_features
    bias_params = out_features if bias else 0
    return weight_params + bias_params

def calculate_se_block_params(channels, reduction=16):
    """SE Block의 파라미터 수 계산"""
    reduced_channels = channels // reduction
    # 첫 번째 FC: channels -> reduced_channels (bias=False)
    fc1_params = calculate_linear_params(channels, reduced_channels, bias=False)
    # 두 번째 FC: reduced_channels -> channels (bias=False)
    fc2_params = calculate_linear_params(reduced_channels, channels, bias=False)
    return fc1_params + fc2_params

def calculate_se_bottleneck_params(in_channels, out_channels, stride=1, reduction=16):
    """SE Bottleneck 블록의 파라미터 수 계산"""
    expansion = 4
    total_params = 0
    
    # Conv1: 1x1 conv (in_channels -> out_channels)
    total_params += calculate_conv2d_params(in_channels, out_channels, 1, bias=False)
    total_params += calculate_batchnorm_params(out_channels)
    
    # Conv2: 3x3 conv (out_channels -> out_channels)
    total_params += calculate_conv2d_params(out_channels, out_channels, 3, bias=False)
    total_params += calculate_batchnorm_params(out_channels)
    
    # Conv3: 1x1 conv (out_channels -> expansion * out_channels)
    expanded_channels = expansion * out_channels
    total_params += calculate_conv2d_params(out_channels, expanded_channels, 1, bias=False)
    total_params += calculate_batchnorm_params(expanded_channels)
    
    # SE Block
    total_params += calculate_se_block_params(expanded_channels, reduction)
    
    # Shortcut connection (if needed)
    if stride != 1 or in_channels != expanded_channels:
        total_params += calculate_conv2d_params(in_channels, expanded_channels, 1, bias=False)
        total_params += calculate_batchnorm_params(expanded_channels)
    
    return total_params

def calculate_se_resnet_params(num_blocks_list, num_classes=10):
    """SE-ResNet의 총 파라미터 수 계산"""
    total_params = 0
    expansion = 4
    
    # Stem (initial conv + bn)
    total_params += calculate_conv2d_params(3, 64, 7, bias=False)  # Conv2d
    total_params += calculate_batchnorm_params(64)  # BatchNorm2d
    
    # Layer 1-4
    in_channels = 64
    layer_channels = [64, 128, 256, 512]
    
    for layer_idx, (num_blocks, out_channels) in enumerate(zip(num_blocks_list, layer_channels)):
        stride = 1 if layer_idx == 0 else 2
        
        for block_idx in range(num_blocks):
            block_stride = stride if block_idx == 0 else 1
            total_params += calculate_se_bottleneck_params(in_channels, out_channels, block_stride)
            in_channels = out_channels * expansion
    
    # Final classifier
    total_params += calculate_linear_params(512 * expansion, num_classes, bias=True)
    
    return total_params

def calculate_flops_estimate(num_blocks_list, input_size=(224, 224)):
    """FLOPs의 대략적인 추정치 계산 (단순화된 버전)"""
    # 이는 매우 단순화된 추정치입니다
    # 실제 FLOPs는 더 복잡한 계산이 필요합니다
    
    total_flops = 0
    h, w = input_size
    
    # Stem
    # Conv2d(3, 64, 7, stride=2) + MaxPool2d(3, stride=2)
    h, w = h // 4, w // 4  # After stem: 56x56
    total_flops += 3 * 64 * 7 * 7 * h * w
    
    # Layers
    channels = [64, 128, 256, 512]
    expansion = 4
    
    for layer_idx, (num_blocks, out_channels) in enumerate(zip(num_blocks_list, channels)):
        if layer_idx > 0:
            h, w = h // 2, w // 2  # Stride 2
        
        for _ in range(num_blocks):
            # Simplified FLOPs for bottleneck block
            # 1x1 conv + 3x3 conv + 1x1 conv + SE block
            in_ch = out_channels * expansion if layer_idx > 0 or _ > 0 else (64 if layer_idx == 0 else channels[layer_idx-1] * expansion)
            
            # 1x1 conv: in_ch -> out_channels
            total_flops += in_ch * out_channels * h * w
            # 3x3 conv: out_channels -> out_channels  
            total_flops += out_channels * out_channels * 9 * h * w
            # 1x1 conv: out_channels -> expansion * out_channels
            total_flops += out_channels * (expansion * out_channels) * h * w
            # SE block (simplified)
            total_flops += (expansion * out_channels) * (expansion * out_channels // 16) * 2
    
    # Final classifier
    total_flops += 512 * expansion * 10
    
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

def save_results_to_files(models_data):
    """결과를 CSV와 텍스트 파일로 저장"""
    
    # CSV 파일로 저장
    csv_filename = './CSV/SE_ResNet_Complexity.csv'
    os.makedirs('./CSV', exist_ok=True)
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model', 'Parameters', 'Parameters_Formatted', 'FLOPs', 'FLOPs_Formatted', 
                     'Model_Size_MB', 'Total_Blocks', 'Layer1_Blocks', 'Layer2_Blocks', 
                     'Layer3_Blocks', 'Layer4_Blocks', 'Stem_Params', 'Residual_Params', 'Classifier_Params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in models_data:
            writer.writerow(data)
    
    # 텍스트 파일로 상세 결과 저장
    txt_filename = './CSV/SE_ResNet_Complexity_Report.txt'
    
    with open(txt_filename, 'w', encoding='utf-8') as txtfile:
        txtfile.write("=" * 90 + "\n")
        txtfile.write("SE-ResNet Models Complexity Analysis (Theoretical Calculation)\n")
        txtfile.write("=" * 90 + "\n")
        txtfile.write(f"{'Model':<15} {'Parameters':<15} {'FLOPs (Est.)':<15} {'Model Size (MB)':<15} {'Blocks':<20}\n")
        txtfile.write("-" * 90 + "\n")
        
        for data in models_data:
            txtfile.write(f"{data['Model']:<15} {data['Parameters_Formatted']:<15} {data['FLOPs_Formatted']:<15} {data['Model_Size_MB']:.2f}MB{'':<7} {[data['Layer1_Blocks'], data['Layer2_Blocks'], data['Layer3_Blocks'], data['Layer4_Blocks']]}\n")
        
        txtfile.write("=" * 90 + "\n\n")
        txtfile.write("Detailed Breakdown:\n")
        txtfile.write("-" * 50 + "\n")
        
        for data in models_data:
            txtfile.write(f"\n{data['Model']}:\n")
            txtfile.write(f"  - Architecture: [{data['Layer1_Blocks']}, {data['Layer2_Blocks']}, {data['Layer3_Blocks']}, {data['Layer4_Blocks']}] blocks per layer\n")
            txtfile.write(f"  - Total layers: {data['Total_Blocks']} residual blocks\n")
            txtfile.write(f"  - SE reduction ratio: 16\n")
            txtfile.write(f"  - Expansion factor: 4 (bottleneck)\n")
            txtfile.write(f"  - Total parameters: {data['Parameters_Formatted']}\n")
            
            total_params = data['Parameters']
            stem_params = data['Stem_Params']
            residual_params = data['Residual_Params']
            classifier_params = data['Classifier_Params']
            
            txtfile.write(f"    * Stem: {format_number(stem_params)} ({stem_params/total_params*100:.1f}%)\n")
            txtfile.write(f"    * Residual blocks: {format_number(residual_params)} ({residual_params/total_params*100:.1f}%)\n")
            txtfile.write(f"    * Classifier: {format_number(classifier_params)} ({classifier_params/total_params*100:.1f}%)\n")
        
        txtfile.write("\n" + "=" * 90 + "\n")
        txtfile.write("Notes:\n")
        txtfile.write("- Parameters: Exact theoretical count\n")
        txtfile.write("- FLOPs: Simplified estimation (actual may vary)\n")
        txtfile.write("- Model Size: Memory requirement assuming float32\n")
        txtfile.write("- SE blocks add ~2-3% parameters compared to standard ResNet\n")
        txtfile.write("=" * 90 + "\n")
    
    return csv_filename, txt_filename

def analyze_se_resnet_complexity():
    """SE-ResNet 모델들의 복합도 분석"""
    
    models = {
        'SE-ResNet-50': [3, 4, 6, 3],
        'SE-ResNet-101': [3, 4, 23, 3], 
        'SE-ResNet-152': [3, 8, 36, 3]
    }
    
    models_data = []
    
    print("=" * 90)
    print("SE-ResNet Models Complexity Analysis (Theoretical Calculation)")
    print("=" * 90)
    print(f"{'Model':<15} {'Parameters':<15} {'FLOPs (Est.)':<15} {'Model Size (MB)':<15} {'Blocks':<20}")
    print("-" * 90)
    
    for name, num_blocks in models.items():
        # 파라미터 수 계산
        total_params = calculate_se_resnet_params(num_blocks)
        
        # FLOPs 추정
        estimated_flops = calculate_flops_estimate(num_blocks)
        
        # 모델 크기 (float32 기준)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # 블록 구성
        blocks_str = f"{num_blocks}"
        
        # 레이어별 파라미터 분석
        stem_params = calculate_conv2d_params(3, 64, 7, bias=False) + calculate_batchnorm_params(64)
        classifier_params = calculate_linear_params(512 * 4, 10, bias=True)
        residual_params = total_params - stem_params - classifier_params
        
        # 데이터 저장
        model_data = {
            'Model': name,
            'Parameters': total_params,
            'Parameters_Formatted': format_number(total_params),
            'FLOPs': estimated_flops,
            'FLOPs_Formatted': format_number(estimated_flops),
            'Model_Size_MB': model_size_mb,
            'Total_Blocks': sum(num_blocks),
            'Layer1_Blocks': num_blocks[0],
            'Layer2_Blocks': num_blocks[1],
            'Layer3_Blocks': num_blocks[2],
            'Layer4_Blocks': num_blocks[3],
            'Stem_Params': stem_params,
            'Residual_Params': residual_params,
            'Classifier_Params': classifier_params
        }
        models_data.append(model_data)
        
        print(f"{name:<15} {format_number(total_params):<15} {format_number(estimated_flops):<15} {model_size_mb:.2f}MB{'':<7} {blocks_str}")
    
    print("=" * 90)
    print("\nDetailed Breakdown:")
    print("-" * 50)
    
    for name, num_blocks in models.items():
        print(f"\n{name}:")
        print(f"  - Architecture: {num_blocks} blocks per layer")
        print(f"  - Total layers: {sum(num_blocks)} residual blocks")
        print(f"  - SE reduction ratio: 16")
        print(f"  - Expansion factor: 4 (bottleneck)")
        
        # 레이어별 파라미터 분석
        total_params = calculate_se_resnet_params(num_blocks)
        print(f"  - Total parameters: {format_number(total_params)}")
        
        # 레이어별 기여도
        stem_params = calculate_conv2d_params(3, 64, 7, bias=False) + calculate_batchnorm_params(64)
        classifier_params = calculate_linear_params(512 * 4, 10, bias=True)
        residual_params = total_params - stem_params - classifier_params
        
        print(f"    * Stem: {format_number(stem_params)} ({stem_params/total_params*100:.1f}%)")
        print(f"    * Residual blocks: {format_number(residual_params)} ({residual_params/total_params*100:.1f}%)")
        print(f"    * Classifier: {format_number(classifier_params)} ({classifier_params/total_params*100:.1f}%)")

    print("\n" + "=" * 90)
    print("Notes:")
    print("- Parameters: Exact theoretical count")
    print("- FLOPs: Simplified estimation (actual may vary)")
    print("- Model Size: Memory requirement assuming float32")
    print("- SE blocks add ~2-3% parameters compared to standard ResNet")
    print("=" * 90)
    
    # 파일로 저장
    csv_file, txt_file = save_results_to_files(models_data)
    print(f"\n결과가 저장되었습니다:")
    print(f"- CSV 파일: {csv_file}")
    print(f"- 텍스트 파일: {txt_file}")
    
    return models_data

if __name__ == '__main__':
    analyze_se_resnet_complexity()