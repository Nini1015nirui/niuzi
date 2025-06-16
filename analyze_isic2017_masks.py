#!/usr/bin/env python3
"""
ISIC2017分割数据集mask阳性像素占比统计脚本
用于评估数据集的类不平衡情况
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def analyze_mask_positive_ratio(mask_dir):
    """
    分析mask目录中所有mask文件的阳性像素占比
    
    Args:
        mask_dir (str): mask文件目录路径
        
    Returns:
        dict: 包含统计信息的字典
    """
    
    mask_path = Path(mask_dir)
    if not mask_path.exists():
        raise ValueError(f"Mask目录不存在: {mask_dir}")
    
    # 获取所有mask文件
    mask_files = list(mask_path.glob("*.png")) + list(mask_path.glob("*.jpg")) + list(mask_path.glob("*.bmp"))
    
    if not mask_files:
        raise ValueError(f"在目录 {mask_dir} 中未找到mask文件")
    
    print(f"找到 {len(mask_files)} 个mask文件")
    
    positive_ratios = []
    file_info = []
    
    # 逐个处理mask文件
    for mask_file in tqdm(mask_files, desc="分析mask文件"):
        try:
            # 读取mask图像
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"警告: 无法读取文件 {mask_file}")
                continue
            
            # 计算阳性像素占比
            total_pixels = mask.shape[0] * mask.shape[1]
            positive_pixels = np.sum(mask > 127)  # 假设>127为阳性像素
            positive_ratio = positive_pixels / total_pixels
            
            positive_ratios.append(positive_ratio)
            file_info.append({
                'filename': mask_file.name,
                'image_size': f"{mask.shape[1]}x{mask.shape[0]}",
                'total_pixels': total_pixels,
                'positive_pixels': positive_pixels,
                'positive_ratio': positive_ratio
            })
            
        except Exception as e:
            print(f"处理文件 {mask_file} 时出错: {e}")
            continue
    
    if not positive_ratios:
        raise ValueError("没有成功处理任何mask文件")
    
    # 转换为numpy数组进行统计
    positive_ratios = np.array(positive_ratios)
    
    # 计算统计信息
    stats = {
        'total_files': len(positive_ratios),
        'mean_ratio': np.mean(positive_ratios),
        'median_ratio': np.median(positive_ratios),
        'min_ratio': np.min(positive_ratios),
        'max_ratio': np.max(positive_ratios),
        'std_ratio': np.std(positive_ratios),
        'percentile_25': np.percentile(positive_ratios, 25),
        'percentile_75': np.percentile(positive_ratios, 75)
    }
    
    return stats, positive_ratios, file_info

def plot_distribution(positive_ratios, save_path=None):
    """
    绘制阳性像素占比分布图
    
    Args:
        positive_ratios (np.array): 阳性像素占比数组
        save_path (str, optional): 保存路径
    """
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 直方图
    plt.subplot(2, 2, 1)
    plt.hist(positive_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('阳性像素占比')
    plt.ylabel('频次')
    plt.title('阳性像素占比分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot(positive_ratios, vert=True)
    plt.ylabel('阳性像素占比')
    plt.title('阳性像素占比箱线图')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 累积分布
    plt.subplot(2, 2, 3)
    sorted_ratios = np.sort(positive_ratios)
    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    plt.plot(sorted_ratios, cumulative, linewidth=2, color='orange')
    plt.xlabel('阳性像素占比')
    plt.ylabel('累积概率')
    plt.title('累积分布函数')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 数据点散布
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(positive_ratios)), positive_ratios, alpha=0.6, s=10)
    plt.xlabel('样本索引')
    plt.ylabel('阳性像素占比')
    plt.title('各样本阳性像素占比')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到: {save_path}")
    
    plt.show()

def save_detailed_report(stats, positive_ratios, file_info, output_dir):
    """
    保存详细分析报告
    
    Args:
        stats (dict): 统计信息
        positive_ratios (np.array): 阳性像素占比数组
        file_info (list): 文件信息列表
        output_dir (str): 输出目录
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存统计摘要
    summary_file = output_path / "mask_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ISIC2017分割数据集Mask阳性像素占比分析报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"总文件数: {stats['total_files']}\n")
        f.write(f"平均阳性像素占比: {stats['mean_ratio']:.4f} ({stats['mean_ratio']*100:.2f}%)\n")
        f.write(f"中位数阳性像素占比: {stats['median_ratio']:.4f} ({stats['median_ratio']*100:.2f}%)\n")
        f.write(f"最小阳性像素占比: {stats['min_ratio']:.4f} ({stats['min_ratio']*100:.2f}%)\n")
        f.write(f"最大阳性像素占比: {stats['max_ratio']:.4f} ({stats['max_ratio']*100:.2f}%)\n")
        f.write(f"标准差: {stats['std_ratio']:.4f}\n")
        f.write(f"25%分位数: {stats['percentile_25']:.4f} ({stats['percentile_25']*100:.2f}%)\n")
        f.write(f"75%分位数: {stats['percentile_75']:.4f} ({stats['percentile_75']*100:.2f}%)\n\n")
        
        # 类不平衡评估
        f.write("类不平衡评估:\n")
        f.write("-" * 20 + "\n")
        imbalance_ratio = (1 - stats['mean_ratio']) / stats['mean_ratio']
        f.write(f"背景:病变像素比例 ≈ {imbalance_ratio:.1f}:1\n")
        
        if stats['mean_ratio'] < 0.1:
            f.write("数据集类不平衡程度: 严重不平衡 (阳性像素<10%)\n")
        elif stats['mean_ratio'] < 0.2:
            f.write("数据集类不平衡程度: 中度不平衡 (阳性像素10-20%)\n")
        elif stats['mean_ratio'] < 0.4:
            f.write("数据集类不平衡程度: 轻度不平衡 (阳性像素20-40%)\n")
        else:
            f.write("数据集类不平衡程度: 相对平衡 (阳性像素>40%)\n")
    
    # 保存详细的文件信息
    df = pd.DataFrame(file_info)
    df = df.sort_values('positive_ratio', ascending=False)
    detail_file = output_path / "mask_analysis_details.csv"
    df.to_csv(detail_file, index=False, encoding='utf-8')
    
    print(f"分析摘要已保存到: {summary_file}")
    print(f"详细信息已保存到: {detail_file}")

def main():
    """主函数"""
    
    # 配置路径
    mask_dirs = [
        "/mnt/d/Spatial-Mamba-main/my_custom_dataset/annotations/training",
        "/mnt/d/Spatial-Mamba-main/my_custom_dataset/annotations/validation",
        "/mnt/d/Spatial-Mamba-main/my_custom_dataset/annotations/test"
    ]
    
    output_dir = "/mnt/d/Spatial-Mamba-main/analysis_results"
    
    all_ratios = []
    all_file_info = []
    total_files = 0
    
    print("开始分析ISIC2017分割数据集...")
    
    # 分析每个子集
    for mask_dir in mask_dirs:
        subset_name = Path(mask_dir).name
        print(f"\n处理子集: {subset_name}")
        
        if not os.path.exists(mask_dir):
            print(f"警告: 目录不存在 {mask_dir}")
            continue
        
        try:
            stats, ratios, file_info = analyze_mask_positive_ratio(mask_dir)
            
            # 添加子集信息
            for info in file_info:
                info['subset'] = subset_name
            
            all_ratios.extend(ratios)
            all_file_info.extend(file_info)
            total_files += stats['total_files']
            
            print(f"{subset_name} 子集统计:")
            print(f"  文件数: {stats['total_files']}")
            print(f"  平均阳性占比: {stats['mean_ratio']:.4f} ({stats['mean_ratio']*100:.2f}%)")
            print(f"  中位数: {stats['median_ratio']:.4f} ({stats['median_ratio']*100:.2f}%)")
            print(f"  范围: {stats['min_ratio']:.4f} - {stats['max_ratio']:.4f}")
            
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
        'percentile_75': np.percentile(all_ratios, 75)
    }
    
    # 输出整体结果
    print("\n" + "="*60)
    print("ISIC2017整体数据集统计结果:")
    print("="*60)
    print(f"总文件数: {overall_stats['total_files']}")
    print(f"平均阳性像素占比: {overall_stats['mean_ratio']:.4f} ({overall_stats['mean_ratio']*100:.2f}%)")
    print(f"中位数阳性像素占比: {overall_stats['median_ratio']:.4f} ({overall_stats['median_ratio']*100:.2f}%)")
    print(f"最小阳性像素占比: {overall_stats['min_ratio']:.4f} ({overall_stats['min_ratio']*100:.2f}%)")
    print(f"最大阳性像素占比: {overall_stats['max_ratio']:.4f} ({overall_stats['max_ratio']*100:.2f}%)")
    print(f"标准差: {overall_stats['std_ratio']:.4f}")
    print(f"25%分位数: {overall_stats['percentile_25']:.4f} ({overall_stats['percentile_25']*100:.2f}%)")
    print(f"75%分位数: {overall_stats['percentile_75']:.4f} ({overall_stats['percentile_75']*100:.2f}%)")
    
    # 类不平衡评估
    print(f"\n类不平衡评估:")
    imbalance_ratio = (1 - overall_stats['mean_ratio']) / overall_stats['mean_ratio']
    print(f"背景:病变像素比例 ≈ {imbalance_ratio:.1f}:1")
    
    # 绘制分布图
    plot_distribution(all_ratios, os.path.join(output_dir, "mask_distribution.png"))
    
    # 保存详细报告
    save_detailed_report(overall_stats, all_ratios, all_file_info, output_dir)
    
    print(f"\n分析完成！结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()