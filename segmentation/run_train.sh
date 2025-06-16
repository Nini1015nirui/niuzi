#!/bin/bash
# ISIC2017分割训练启动脚本

# 激活conda环境
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate niuzi

# 设置Python路径
export PYTHONPATH="/mnt/d/Spatial-Mamba-main/segmentation:$PYTHONPATH"

# 切换到工作目录
cd /mnt/d/Spatial-Mamba-main/segmentation

# 启动训练
python tools/train.py configs/spatialmamba/upernet_spatialmamba_4xb4-160k_ade20k-512x512_tiny.py --work-dir work_dirs/upernet_spatialmamba_tiny_isic2017