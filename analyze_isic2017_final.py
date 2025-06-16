#!/usr/bin/env python3
"""
ISIC2017分割数据集mask阳性像素占比统计脚本（最终版）
正确处理0-1二值标签格式
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def analyze_mask_positive_ratio(mask_dir):
    """
    分析mask目录中所有mask文件的阳性像素占比
    处理0-1二值标签格式
    """
    
    mask_path = Path(mask_dir)
    if not mask_path.exists():
        raise ValueError(f"Mask目录不存在: {mask_dir}")
    
    mask_files = list(mask_path.glob("*.png"))
    
    if not mask_files:
        raise ValueError(f"在目录 {mask_dir} 中未找到mask文件")
    
    print(f"找到 {len(mask_files)} 个mask文件")
    
    positive_ratios = []
    file_info = []
    
    for mask_file in tqdm(mask_files, desc="分析mask文件"):
        try:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"警告: 无法读取文件 {mask_file}")
                continue
            
            total_pixels = mask.shape[0] * mask.shape[1]
            # 对于0-1标签，直接统计值为1的像素
            positive_pixels = np.sum(mask == 1)
            positive_ratio = positive_pixels / total_pixels
            
            positive_ratios.append(positive_ratio)
            file_info.append({
                'filename': mask_file.name,
                'image_size': f"{mask.shape[1]}x{mask.shape[0]}",
                'total_pixels': total_pixels,
                'positive_pixels': positive_pixels,
                'positive_ratio': positive_ratio,
                'unique_values': str(np.unique(mask).tolist())
            })
            
        except Exception as e:
            print(f"处理文件 {mask_file} 时出错: {e}")
            continue
    
    if not positive_ratios:
        raise ValueError("没有成功处理任何mask文件")
    
    positive_ratios = np.array(positive_ratios)
    
    stats = {
        'total_files': len(positive_ratios),
        'mean_ratio': np.mean(positive_ratios),
        'median_ratio': np.median(positive_ratios),
        'min_ratio': np.min(positive_ratios),
        'max_ratio': np.max(positive_ratios),
        'std_ratio': np.std(positive_ratios),
        'percentile_25': np.percentile(positive_ratios, 25),
        'percentile_75': np.percentile(positive_ratios, 75),
        'zero_ratio_count': np.sum(positive_ratios == 0),
        'nonzero_ratio_count': np.sum(positive_ratios > 0)
    }
    
    return stats, positive_ratios, file_info

def main():
    """主函数"""
    
    mask_dirs = [
        "/mnt/d/Spatial-Mamba-main/my_custom_dataset/annotations/training",
        "/mnt/d/Spatial-Mamba-main/my_custom_dataset/annotations/validation"
    ]
    
    all_ratios = []
    all_file_info = []
    
    print("开始分析ISIC2017分割数据集（0-1标签格式）...")
    
    for mask_dir in mask_dirs:
        subset_name = Path(mask_dir).name
        print(f"\n处理子集: {subset_name}")
        
        if not os.path.exists(mask_dir):
            print(f"警告: 目录不存在 {mask_dir}")
            continue
        
        try:
            stats, ratios, file_info = analyze_mask_positive_ratio(mask_dir)
            
            for info in file_info:
                info['subset'] = subset_name
            
            all_ratios.extend(ratios)
            all_file_info.extend(file_info)
            
            print(f"{subset_name} 子集统计:")
            print(f"  文件数: {stats['total_files']}")
            print(f"  平均阳性占比: {stats['mean_ratio']:.6f} ({stats['mean_ratio']*100:.4f}%)")
            print(f"  中位数: {stats['median_ratio']:.6f} ({stats['median_ratio']*100:.4f}%)")
            print(f"  范围: {stats['min_ratio']:.6f} - {stats['max_ratio']:.6f}")
            print(f"  无病变样本: {stats['zero_ratio_count']}")
            print(f"  有病变样本: {stats['nonzero_ratio_count']}")
            
        except Exception as e:
            print(f"处理 {mask_dir} 时出错: {e}")
            continue
    
    if not all_ratios:
        print("错误: 没有找到任何有效的mask文件")
        return
    
    # 计算整体统计
    all_ratios = np.array(all_ratios)
    overall_stats = {
        'total_files': len(all_ratios),
        'mean_ratio': np.mean(all_ratios),
        'median_ratio': np.median(all_ratios),
        'min_ratio': np.min(all_ratios),
        'max_ratio': np.max(all_ratios),
        'std_ratio': np.std(all_ratios),
        'percentile_25': np.percentile(all_ratios, 25),
        'percentile_75': np.percentile(all_ratios, 75),
        'zero_ratio_count': np.sum(all_ratios == 0),
        'nonzero_ratio_count': np.sum(all_ratios > 0)
    }
    
    # 输出结果
    print("\n" + "="*70)
    print("ISIC2017整体数据集统计结果:")
    print("="*70)
    print(f"总文件数: {overall_stats['total_files']}")
    print(f"平均阳性像素占比: {overall_stats['mean_ratio']:.6f} ({overall_stats['mean_ratio']*100:.4f}%)")
    print(f"中位数阳性像素占比: {overall_stats['median_ratio']:.6f} ({overall_stats['median_ratio']*100:.4f}%)")
    print(f"最小阳性像素占比: {overall_stats['min_ratio']:.6f} ({overall_stats['min_ratio']*100:.4f}%)")
    print(f"最大阳性像素占比: {overall_stats['max_ratio']:.6f} ({overall_stats['max_ratio']*100:.4f}%)")
    print(f"标准差: {overall_stats['std_ratio']:.6f}")
    print(f"25%分位数: {overall_stats['percentile_25']:.6f} ({overall_stats['percentile_25']*100:.4f}%)")
    print(f"75%分位数: {overall_stats['percentile_75']:.6f} ({overall_stats['percentile_75']*100:.4f}%)")
    print(f"无病变样本数: {overall_stats['zero_ratio_count']}")
    print(f"有病变样本数: {overall_stats['nonzero_ratio_count']}")
    
    # 类不平衡评估
    print(f"\n类不平衡评估:")
    if overall_stats['mean_ratio'] > 0:
        imbalance_ratio = (1 - overall_stats['mean_ratio']) / overall_stats['mean_ratio']
        print(f"背景:病变像素比例 ≈ {imbalance_ratio:.1f}:1")
        
        if overall_stats['mean_ratio'] < 0.01:
            level = "极度不平衡"
        elif overall_stats['mean_ratio'] < 0.05:
            level = "严重不平衡"
        elif overall_stats['mean_ratio'] < 0.1:
            level = "中度不平衡"
        elif overall_stats['mean_ratio'] < 0.3:
            level = "轻度不平衡"
        else:
            level = "相对平衡"
        
        print(f"数据集类不平衡程度: {level} (阳性像素{overall_stats['mean_ratio']*100:.4f}%)")
    else:
        print("所有样本均无病变像素")
    
    # 显示一些样本统计
    if overall_stats['nonzero_ratio_count'] > 0:
        print(f"\n样本分布:")
        print(f"病变样本占比: {overall_stats['nonzero_ratio_count']/overall_stats['total_files']*100:.2f}%")
        print(f"最大病变占比: {overall_stats['max_ratio']*100:.4f}%")
        
        # 显示病变占比分布
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        bin_labels = ["0%", "0-1%", "1-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
        
        print(f"\n病变占比分布:")
        for i in range(len(bins)-1):
            count = np.sum((all_ratios >= bins[i]) & (all_ratios < bins[i+1]))
            if i == len(bins)-2:  # 最后一个区间包含上界
                count = np.sum((all_ratios >= bins[i]) & (all_ratios <= bins[i+1]))
            print(f"  {bin_labels[i+1]}: {count} 个样本 ({count/overall_stats['total_files']*100:.2f}%)")

if __name__ == "__main__":
    main()