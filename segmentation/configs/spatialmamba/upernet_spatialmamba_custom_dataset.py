# Custom Dataset Training Configuration for Spatial-Mamba
# 基于你的自定义数据集的Spatial-Mamba分割训练配置

_base_ = [
    '../_base_/models/upernet_r50.py',  # 基础模型配置
    '../_base_/datasets/custom_dataset.py',  # 你的数据集配置
    '../_base_/default_runtime.py',    # 运行时配置
    '../_base_/schedules/schedule_160k.py'  # 训练策略
]

# =====================================================
# 模型配置 - Spatial-Mamba
# =====================================================
model = dict(
    backbone=dict(
        _delete_=True,  # 删除原有backbone配置
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained="",  # 预训练权重路径，留空表示随机初始化
        
        # 模型尺寸配置（可选择：tiny, small, base）
        dims=64,        # tiny: 64, small: 96, base: 128
        d_state=1,      # Mamba状态维度
        depths=(2, 4, 8, 4),  # 各层深度
        drop_path_rate=0.2,   # Drop path比率
    ),
    
    # 解码头配置 - ISIC2017二分类分割
    decode_head=dict(
        in_channels=[64, 128, 256, 512],  # 对应backbone输出通道
        num_classes=2,  # 🔥 ISIC2017二分类：背景(0) + 皮肤病变(1)
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,  # 二分类使用标准交叉熵
            class_weight=[1.0, 2.0],  # 病变类别权重更高
            avg_non_ignore=True
        )
    ),
    
    auxiliary_head=dict(
        in_channels=256,
        num_classes=2,  # 🔥 与decode_head保持一致
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, 2.0],
            avg_non_ignore=True
        )
    )
)

# =====================================================
# 数据集配置覆盖
# =====================================================
# ISIC2017皮肤病变分割数据集路径
data_root = '../my_custom_dataset'  # 🔥 ISIC2017数据集路径

train_dataloader = dict(
    batch_size=2,  # 🔥 根据GPU内存调整（RTX 4060建议2-4）
    dataset=dict(
        data_root=data_root,
    ))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
    ))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
    ))

# =====================================================
# 训练配置调整
# =====================================================
# 训练迭代数（根据数据集大小调整）
train_cfg = dict(
    max_iters=40000,    # 🔥 根据数据集大小调整
    val_interval=2000   # 每2000次迭代验证一次
)

# 学习率策略
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,  # 与max_iters保持一致
        by_epoch=False,
    )
]

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006,  # 🔥 学习率，可根据需要调整
        betas=(0.9, 0.999), 
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        })
)

# =====================================================
# 日志和检查点配置
# =====================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=2000,  # 每2000次迭代保存检查点
        max_keep_ckpts=3),  # 最多保留3个检查点
    logger=dict(
        type='LoggerHook', 
        interval=50,  # 每50次迭代打印日志
        log_metric_by_epoch=False),
)

# 工作目录
work_dir = './work_dirs/isic2017_spatialmamba_training'

# =====================================================
# 使用说明
# =====================================================
"""
ISIC2017皮肤病变分割训练配置

数据集信息：
- ISIC2017皮肤病变分割数据集
- 2000张皮肤镜图像
- 二分类任务：背景(0) + 皮肤病变(1)
- 数据路径：../my_custom_dataset

训练命令：
   cd segmentation
   python tools/train.py configs/spatialmamba/upernet_spatialmamba_custom_dataset.py

监控训练：
   tensorboard --logdir work_dirs/isic2017_spatialmamba_training

测试模型：
   python tools/test.py configs/spatialmamba/upernet_spatialmamba_custom_dataset.py work_dirs/isic2017_spatialmamba_training/latest.pth
"""