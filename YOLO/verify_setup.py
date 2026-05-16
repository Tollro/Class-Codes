#!/usr/bin/env python
"""验证SSD训练脚本的配置和导入"""

import sys
from pathlib import Path

print("=" * 80)
print("SSD 训练脚本验证")
print("=" * 80)

# 检查必要的文件
required_files = [
    'SSD/config.json',
    'SSD/SSD_Train.py',
    'SSD/data_loader.py',
    'SSD/metrics.py',
    'SSD/utils.py'
]

print("\n1. 检查文件完整性...")
all_files_exist = True
for file in required_files:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n✗ 某些文件缺失！")
    sys.exit(1)

# 检查配置文件内容
print("\n2. 检查配置文件...")
try:
    import json
    with open('SSD/config.json', 'r') as f:
        config = json.load(f)
    
    required_keys = ['data_dir', 'num_classes', 'training', 'validation']
    missing_keys = []
    
    for key in required_keys:
        if key in config:
            print(f"   ✓ {key}")
        else:
            print(f"   ✗ {key} (缺失)")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n✗ 配置文件缺失关键字段: {missing_keys}")
        sys.exit(1)
    
    print(f"\n   数据集路径: {config['data_dir']}")
    print(f"   类别数: {config['num_classes']}")
    print(f"   训练参数: epochs={config['training']['epochs']}, batch_size={config['training']['batch_size']}")
    
except Exception as e:
    print(f"\n✗ 配置文件读取失败: {e}")
    sys.exit(1)

# 检查导入
print("\n3. 检查Python依赖...")
dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'Torchvision',
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'tqdm': 'tqdm'
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} (未安装)")
        missing_deps.append(name)

if missing_deps:
    print(f"\n✗ 缺失依赖: {', '.join(missing_deps)}")
    print("请运行: pip install -r SSD/requirements.txt")
    sys.exit(1)

# 检查自定义模块导入
print("\n4. 检查自定义模块...")
sys.path.insert(0, 'SSD')

try:
    from data_loader import FruitDataset, collate_fn
    print("   ✓ data_loader.py")
except Exception as e:
    print(f"   ✗ data_loader.py: {e}")
    sys.exit(1)

try:
    from metrics import MetricsCalculator
    print("   ✓ metrics.py")
except Exception as e:
    print(f"   ✗ metrics.py: {e}")
    sys.exit(1)

try:
    from utils import setup_logger, plot_metrics
    print("   ✓ utils.py")
except Exception as e:
    print(f"   ✗ utils.py: {e}")
    sys.exit(1)

try:
    from SSD_Train import SSDTrainer
    print("   ✓ SSD_Train.py")
except Exception as e:
    print(f"   ✗ SSD_Train.py: {e}")
    print(f"   错误详情: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ 所有检查通过！可以开始训练。")
print("=" * 80)

print("\n快速开始:")
print("  python SSD/quick_start.py train")
print("\n或者直接运行:")
print("  python SSD/SSD_Train.py")
