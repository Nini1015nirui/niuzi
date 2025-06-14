#!/usr/bin/env python3
"""
Dataset Format Converter for Spatial-Mamba Segmentation
数据集格式转换工具

支持的转换格式：
1. COCO格式 -> ADE20K格式
2. VOC格式 -> ADE20K格式  
3. 单独的RGB彩色标注 -> 灰度索引标注
4. 其他常见格式
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import shutil
from tqdm import tqdm


class DatasetConverter:
    """数据集格式转换器"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出目录结构
        for split in ['training', 'validation']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)
    
    def convert_voc_format(self, train_ratio=0.8):
        """
        转换VOC格式数据集
        
        期望输入结构：
        input_dir/
        ├── JPEGImages/     # 原始图像
        ├── SegmentationClass/  # 分割标注
        └── ImageSets/
            └── Segmentation/
                ├── train.txt
                └── val.txt
        """
        print("🔄 转换VOC格式数据集...")
        
        jpeg_dir = self.input_dir / 'JPEGImages'
        seg_dir = self.input_dir / 'SegmentationClass' 
        sets_dir = self.input_dir / 'ImageSets' / 'Segmentation'
        
        # 检查目录是否存在
        if not all([jpeg_dir.exists(), seg_dir.exists()]):
            print("❌ VOC格式目录不完整")
            return False
        
        # 读取分割列表
        train_list = []
        val_list = []
        
        if (sets_dir / 'train.txt').exists():
            with open(sets_dir / 'train.txt', 'r') as f:
                train_list = [line.strip() for line in f.readlines()]
        
        if (sets_dir / 'val.txt').exists():
            with open(sets_dir / 'val.txt', 'r') as f:
                val_list = [line.strip() for line in f.readlines()]
        
        # 如果没有分割列表，自动创建
        if not train_list and not val_list:
            all_images = [f.stem for f in jpeg_dir.glob('*.jpg')]
            split_idx = int(len(all_images) * train_ratio)
            train_list = all_images[:split_idx]
            val_list = all_images[split_idx:]
            print(f"📋 自动分割: 训练集{len(train_list)}张, 验证集{len(val_list)}张")
        
        # 转换训练集
        self._convert_voc_split(jpeg_dir, seg_dir, train_list, 'training')
        
        # 转换验证集
        self._convert_voc_split(jpeg_dir, seg_dir, val_list, 'validation')
        
        print("✅ VOC格式转换完成")
        return True
    
    def _convert_voc_split(self, jpeg_dir, seg_dir, file_list, split):
        """转换VOC单个分割"""
        output_img_dir = self.output_dir / 'images' / split
        output_ann_dir = self.output_dir / 'annotations' / split
        
        for filename in tqdm(file_list, desc=f"转换{split}集"):
            # 复制图像
            img_src = jpeg_dir / f"{filename}.jpg"
            if not img_src.exists():
                img_src = jpeg_dir / f"{filename}.png"
            
            if img_src.exists():
                img_dst = output_img_dir / f"{filename}.jpg"
                shutil.copy2(img_src, img_dst)
            
            # 转换标注
            ann_src = seg_dir / f"{filename}.png"
            if ann_src.exists():
                ann_dst = output_ann_dir / f"{filename}.png"
                self._convert_annotation(ann_src, ann_dst)
    
    def convert_rgb_annotations(self, class_colors, train_ratio=0.8):
        """
        转换RGB彩色标注为灰度索引标注
        
        Args:
            class_colors: 类别颜色映射 {class_id: [R, G, B]}
            train_ratio: 训练集比例
        """
        print("🔄 转换RGB彩色标注...")
        
        # 假设输入目录包含images和annotations子目录
        input_img_dir = self.input_dir / 'images'
        input_ann_dir = self.input_dir / 'annotations'
        
        if not input_img_dir.exists() or not input_ann_dir.exists():
            print("❌ 找不到images或annotations目录")
            return False
        
        # 获取所有图像文件
        img_files = list(input_img_dir.glob('*.jpg')) + list(input_img_dir.glob('*.png'))
        
        # 分割训练集和验证集
        split_idx = int(len(img_files) * train_ratio)
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]
        
        # 转换训练集
        self._convert_rgb_split(input_img_dir, input_ann_dir, train_files, 'training', class_colors)
        
        # 转换验证集
        self._convert_rgb_split(input_img_dir, input_ann_dir, val_files, 'validation', class_colors)
        
        print("✅ RGB标注转换完成")
        return True
    
    def _convert_rgb_split(self, input_img_dir, input_ann_dir, file_list, split, class_colors):
        """转换RGB标注单个分割"""
        output_img_dir = self.output_dir / 'images' / split
        output_ann_dir = self.output_dir / 'annotations' / split
        
        for img_file in tqdm(file_list, desc=f"转换{split}集"):
            # 复制图像
            img_dst = output_img_dir / img_file.name
            shutil.copy2(img_file, img_dst)
            
            # 转换标注
            ann_file = input_ann_dir / (img_file.stem + '.png')
            if ann_file.exists():
                ann_dst = output_ann_dir / (img_file.stem + '.png')
                self._convert_rgb_to_index(ann_file, ann_dst, class_colors)
    
    def _convert_rgb_to_index(self, rgb_path, output_path, class_colors):
        """将RGB彩色标注转换为灰度索引标注"""
        # 读取RGB标注
        rgb_img = cv2.imread(str(rgb_path))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 创建索引标注
        h, w = rgb_img.shape[:2]
        index_img = np.zeros((h, w), dtype=np.uint8)
        
        # 为每个类别分配像素
        for class_id, color in class_colors.items():
            # 创建颜色掩码
            mask = np.all(rgb_img == color, axis=2)
            index_img[mask] = class_id
        
        # 保存索引标注
        cv2.imwrite(str(output_path), index_img)
    
    def _convert_annotation(self, src_path, dst_path):
        """转换单个标注文件（通用方法）"""
        # 读取源标注
        ann = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        
        if ann is None:
            print(f"⚠️ 无法读取标注文件: {src_path}")
            return
        
        # 直接保存（如果已经是索引格式）
        cv2.imwrite(str(dst_path), ann)
    
    def convert_coco_format(self, annotation_file, train_ratio=0.8):
        """
        转换COCO格式数据集
        
        Args:
            annotation_file: COCO标注JSON文件路径
            train_ratio: 训练集比例
        """
        print("🔄 转换COCO格式数据集...")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # 这里需要根据具体的COCO格式实现转换逻辑
        # 由于COCO格式比较复杂，这里提供基本框架
        print("⚠️ COCO格式转换需要根据具体情况实现")
        return False
    
    def create_data_info(self):
        """创建数据集信息文件"""
        print("📊 生成数据集信息...")
        
        info = {
            'name': self.output_dir.name,
            'splits': {},
            'classes': set()
        }
        
        for split in ['training', 'validation']:
            ann_dir = self.output_dir / 'annotations' / split
            img_dir = self.output_dir / 'images' / split
            
            ann_files = list(ann_dir.glob('*.png'))
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            
            info['splits'][split] = {
                'num_images': len(img_files),
                'num_annotations': len(ann_files)
            }
            
            # 统计类别
            for ann_file in ann_files:
                ann = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
                if ann is not None:
                    unique_classes = np.unique(ann)
                    info['classes'].update(unique_classes.tolist())
        
        # 转换set为list
        info['classes'] = sorted(list(info['classes']))
        info['num_classes'] = len(info['classes'])
        
        # 保存信息文件
        info_file = self.output_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📄 数据集信息已保存到: {info_file}")
        
        return info


def main():
    parser = argparse.ArgumentParser(description='数据集格式转换工具')
    parser.add_argument('input_dir', help='输入数据集目录')
    parser.add_argument('output_dir', help='输出数据集目录') 
    parser.add_argument('--format', choices=['voc', 'rgb', 'coco'], 
                       required=True, help='输入数据集格式')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--class-colors', 
                       help='RGB格式的类别颜色映射JSON文件')
    
    args = parser.parse_args()
    
    converter = DatasetConverter(args.input_dir, args.output_dir)
    
    success = False
    
    if args.format == 'voc':
        success = converter.convert_voc_format(args.train_ratio)
    
    elif args.format == 'rgb':
        if not args.class_colors:
            print("❌ RGB格式需要提供--class-colors参数")
            return
        
        with open(args.class_colors, 'r') as f:
            class_colors = json.load(f)
        
        # 转换字符串键为整数
        class_colors = {int(k): v for k, v in class_colors.items()}
        
        success = converter.convert_rgb_annotations(class_colors, args.train_ratio)
    
    elif args.format == 'coco':
        print("❌ COCO格式转换尚未实现")
        return
    
    if success:
        converter.create_data_info()
        print("\n🎉 数据集转换完成!")
        print(f"📁 输出目录: {args.output_dir}")
        print("\n📝 下一步:")
        print("1. 运行数据集验证工具")
        print("2. 调整训练配置文件")
        print("3. 开始训练")


if __name__ == '__main__':
    main()