# ISIC2017 皮肤病变分割配置 - Spatial-Mamba
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/default_runtime.py'
]

# 数据配置
dataset_type = 'ADE20KDataset'
data_root = '../my_custom_dataset'  # 从segmentation目录的相对路径
crop_size = (512, 512)
backend_args = None

# 训练数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomFlip', direction='vertical', prob=0.3),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=8),
    dict(type='PackSegInputs')
]

# 测试数据管道
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# 训练数据集
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline,
        reduce_zero_label=False))

# 验证数据集
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=test_pipeline,
        reduce_zero_label=False,
        backend_args=backend_args))

# 测试数据集
test_dataloader = val_dataloader

# 🔧 关键修复：仅在backbone中加载预训练权重，分类头随机初始化
# load_from = 'pretrained_weights/upernet_spatialmamba_4xb4-160k_ade20k-512x512_tiny_iter_144000.pth'  # 禁用顶层加载

# 模型配置
model = dict(
    pretrained=None,  # 明确禁用基础配置的ResNet50权重下载
    backbone=dict(
        _delete_=True,
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        dims=64,
        depths=(2, 4, 8, 4),
        drop_path_rate=0.1,
        recompute=True,  # 启用梯度检查点，减少显存使用约60%
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained_weights/upernet_spatialmamba_4xb4-160k_ade20k-512x512_tiny_iter_144000.pth',
            prefix='backbone.',  # 🔧 仅加载backbone部分，完全忽略分割头
            strict=False,  # 允许部分权重不匹配（分割头不加载）
            map_location='cpu',  # 避免内存问题
            # 仅使用Spatial-Mamba backbone的预训练权重，分类头从零开始训练
        ),
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=1,  # 二分类分割推荐使用1个输出通道
        threshold=0.5,  # 🔧 显式指定二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 分类头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,  # 二分类使用sigmoid
            class_weight=[2.0],  # 病变类别权重
            avg_non_ignore=True
        )
    ),
    auxiliary_head=dict(
        in_channels=256,  # 来自backbone第3层的输出
        num_classes=1,
        threshold=0.5,  # 🔧 显式指定二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 辅助头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=[2.0],
            avg_non_ignore=True
        )
    ),
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size
    )
)

# 数据增强管道 (针对医学图像优化)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.3, direction='vertical'),  # 医学图像可垂直翻转
    dict(type='PhotoMetricDistortion', 
         brightness_delta=16,
         contrast_range=(0.9, 1.1),
         saturation_range=(0.9, 1.1),
         hue_delta=8),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# 数据加载
train_dataloader = dict(
    batch_size=2,  # 适合大图像
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'
        ),
        pipeline=train_pipeline,
        reduce_zero_label=False  # 二分类保持原始标签
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'
        ),
        pipeline=test_pipeline,
        reduce_zero_label=False  # 二分类保持原始标签
    )
)

test_dataloader = val_dataloader

# 训练配置 (针对ISIC2017优化)
max_iters = 50  # 适合2000张医学图像的迭代数
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=10  # 每1000次迭代验证
)

# 学习率调度 (快速测试版本 - 短期训练)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=5),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=5, end=max_iters, by_epoch=False)
]

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00003,  # 较小学习率适合医学图像
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# 评估指标 (二分类医学分割专用)
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],  # 只计算mIoU和mDice
    nan_to_num=0,
    threshold=0.5,  # 二分类阈值
    collect_device='cpu',
    prefix='',
    format_only=False,
    keep_results=False
)
test_evaluator = val_evaluator

# 训练循环配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=10  # 每10次迭代验证
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 日志和检查点配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10,  # 每2000次迭代保存
        max_keep_ckpts=5
    ),
    logger=dict(
        type='LoggerHook',
        interval=10,  # 每100次迭代打印日志
        log_metric_by_epoch=False
    )
)

# 轮次和执行器配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=10
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录
work_dir = './work_dirs/isic2017_quick_test'

# ISIC2017数据集信息注释
"""
ISIC2017皮肤病变分割数据集配置说明：

数据集特点：
- 总计2000张皮肤镜图像
- 训练集：1600张 | 验证集：400张
- 任务：二分类分割 (背景=0, 皮肤病变=1)
- 图像尺寸：变化，通常767x1022

医学图像分割优化：
- 使用类别权重 [1.0, 2.0] 处理不平衡问题
- Dice系数评估（医学标准指标）
- 适合医学图像的数据增强
- 较小学习率和稳定训练策略

训练命令：
  cd segmentation
  PYTHONPATH=. python tools/train.py configs/spatialmamba/isic2017_segmentation.py

测试命令：
  PYTHONPATH=. python tools/test.py configs/spatialmamba/isic2017_segmentation.py work_dirs/isic2017_segmentation/latest.pth
"""