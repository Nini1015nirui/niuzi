#!/usr/bin/env python3
"""
Custom Dataset Preparation and Validation Tool for Spatial-Mamba Segmentation
自定义数据集准备和验证工具

功能：
1. 验证数据集格式
2. 生成类别统计信息
3. 检查图像和标注对应关系
4. 可视化数据样本
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter
import argparse


class CustomDatasetValidator:
    """自定义数据集验证器"""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.stats = {
            'train_images': 0,
            'val_images': 0,
            'train_annotations': 0,
            'val_annotations': 0,
            'classes': set(),
            'errors': []
        }
    
    def validate_structure(self):
        """验证数据集文件夹结构"""
        print("🔍 验证数据集结构...")
        
        required_dirs = [
            'images/training',
            'images/validation', 
            'annotations/training',
            'annotations/validation'
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                self.stats['errors'].append(f"缺少目录: {dir_path}")
                print(f"❌ 缺少目录: {full_path}")
            else:
                print(f"✅ 目录存在: {dir_path}")
        
        return len(self.stats['errors']) == 0
    
    def validate_files(self):
        """验证文件对应关系和格式"""
        print("\n🔍 验证文件对应关系...")
        
        # 检查训练集
        self._validate_split('training')
        # 检查验证集
        self._validate_split('validation')
        
        return len(self.stats['errors']) == 0
    
    def _validate_split(self, split):
        """验证单个数据分割"""
        img_dir = self.dataset_root / 'images' / split
        ann_dir = self.dataset_root / 'annotations' / split
        
        # 获取图像文件
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(img_dir.glob(ext))
        
        # 获取标注文件
        ann_files = list(ann_dir.glob('*.png'))
        
        self.stats[f'{split}_images'] = len(img_files)
        self.stats[f'{split}_annotations'] = len(ann_files)
        
        print(f"  {split}: {len(img_files)} 图像, {len(ann_files)} 标注")
        
        # 检查对应关系
        for img_file in img_files:
            # 构造对应的标注文件名
            ann_file = ann_dir / (img_file.stem + '.png')
            
            if not ann_file.exists():
                error = f"{split}: 图像 {img_file.name} 缺少对应标注"
                self.stats['errors'].append(error)
                print(f"❌ {error}")
                continue
            
            # 验证标注文件
            try:
                ann = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
                if ann is None:
                    error = f"{split}: 无法读取标注文件 {ann_file.name}"
                    self.stats['errors'].append(error)
                    print(f"❌ {error}")
                    continue
                
                # 统计类别
                unique_labels = np.unique(ann)
                self.stats['classes'].update(unique_labels.tolist())
                
            except Exception as e:
                error = f"{split}: 标注文件 {ann_file.name} 错误: {str(e)}"
                self.stats['errors'].append(error)
                print(f"❌ {error}")
    
    def analyze_classes(self):
        """分析类别分布"""
        print("\n📊 分析类别分布...")
        
        class_counts = Counter()
        
        for split in ['training', 'validation']:
            ann_dir = self.dataset_root / 'annotations' / split
            ann_files = list(ann_dir.glob('*.png'))
            
            for ann_file in ann_files:
                try:
                    ann = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
                    if ann is not None:
                        unique, counts = np.unique(ann, return_counts=True)
                        for cls, count in zip(unique, counts):
                            class_counts[cls] += count
                except:
                    continue
        
        print(f"发现 {len(class_counts)} 个类别:")
        for cls in sorted(class_counts.keys()):
            print(f"  类别 {cls}: {class_counts[cls]:,} 像素")
        
        # 保存类别信息
        class_info = {
            'num_classes': len(class_counts),
            'class_names': [f'class_{i}' for i in sorted(class_counts.keys())],
            'class_pixel_counts': dict(class_counts)
        }
        
        info_file = self.dataset_root / 'dataset_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(class_info, f, indent=2, ensure_ascii=False)
        
        print(f"📄 类别信息已保存到: {info_file}")
        
        return class_info
    
    def visualize_samples(self, num_samples=3):
        """可视化数据样本"""
        print(f"\n🖼️ 可视化 {num_samples} 个样本...")
        
        # 创建可视化目录
        vis_dir = self.dataset_root / 'visualization'
        vis_dir.mkdir(exist_ok=True)
        
        # 从训练集选择样本
        img_dir = self.dataset_root / 'images' / 'training'
        ann_dir = self.dataset_root / 'annotations' / 'training'
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for i, img_file in enumerate(img_files[:num_samples]):
            ann_file = ann_dir / (img_file.stem + '.png')
            
            if not ann_file.exists():
                continue
            
            # 读取图像和标注
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ann = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原图
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 标注图
            axes[1].imshow(ann, cmap='tab20')
            axes[1].set_title('Annotation')
            axes[1].axis('off')
            
            # 叠加图
            overlay = img.copy()
            colored_ann = plt.cm.tab20(ann / ann.max())[:, :, :3]
            overlay = (overlay * 0.6 + colored_ann * 255 * 0.4).astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            vis_file = vis_dir / f'sample_{i+1}.png'
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  已保存可视化: {vis_file}")
    
    def generate_config_template(self, num_classes):
        """生成配置文件模板"""
        print(f"\n📝 生成配置文件模板...")
        
        config_template = f'''# 自动生成的配置文件模板
# 请根据实际情况调整参数

_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/custom_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# 数据集路径（相对于segmentation目录）
data_root = '{self.dataset_root.relative_to(Path.cwd())}'

# 模型配置
model = dict(
    backbone=dict(
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        dims=64,  # tiny: 64, small: 96, base: 128
        d_state=1,
        depths=(2, 4, 8, 4),
        drop_path_rate=0.2,
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes={num_classes},  # 包括背景类
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes={num_classes},
    )
)

# 数据加载配置
train_dataloader = dict(
    batch_size=2,  # 根据GPU内存调整
    dataset=dict(data_root=data_root))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root))

# 训练配置
train_cfg = dict(
    max_iters={max(20000, self.stats['train_images'] * 100)},  # 根据数据量调整
    val_interval=2000
)

# 工作目录
work_dir = './work_dirs/{self.dataset_root.name}_training'
'''
        
        config_file = self.dataset_root / 'suggested_config.py'
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_template)
        
        print(f"📄 配置文件模板已保存到: {config_file}")
    
    def run_validation(self):
        """运行完整验证流程"""
        print("🚀 开始验证自定义数据集...")
        print(f"📁 数据集路径: {self.dataset_root}")
        
        # 1. 验证结构
        if not self.validate_structure():
            print("\n❌ 数据集结构验证失败!")
            return False
        
        # 2. 验证文件
        if not self.validate_files():
            print("\n❌ 文件验证失败!")
            return False
        
        # 3. 分析类别
        class_info = self.analyze_classes()
        
        # 4. 可视化样本
        self.visualize_samples()
        
        # 5. 生成配置模板
        self.generate_config_template(class_info['num_classes'])
        
        # 6. 输出总结
        print("\n📋 验证总结:")
        print(f"  ✅ 训练图像: {self.stats['train_images']}")
        print(f"  ✅ 验证图像: {self.stats['val_images']}")
        print(f"  ✅ 类别数量: {class_info['num_classes']}")
        print(f"  ✅ 错误数量: {len(self.stats['errors'])}")
        
        if len(self.stats['errors']) == 0:
            print("\n🎉 数据集验证通过! 可以开始训练了!")
            print("\n📝 下一步操作:")
            print(f"1. 复制生成的配置文件到 configs/spatialmamba/")
            print(f"2. 根据需要调整配置参数")
            print(f"3. 开始训练: python tools/train.py configs/spatialmamba/your_config.py")
        else:
            print("\n⚠️ 发现以下错误，请修复后重新验证:")
            for error in self.stats['errors']:
                print(f"   - {error}")
        
        return len(self.stats['errors']) == 0


def main():
    parser = argparse.ArgumentParser(description='验证和准备自定义分割数据集')
    parser.add_argument('dataset_path', help='数据集根目录路径')
    parser.add_argument('--samples', type=int, default=3, help='可视化样本数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ 数据集路径不存在: {args.dataset_path}")
        return
    
    validator = CustomDatasetValidator(args.dataset_path)
    validator.run_validation()


if __name__ == '__main__':
    main()