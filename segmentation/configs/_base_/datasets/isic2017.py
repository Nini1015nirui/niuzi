# ISIC2017皮肤病变分割数据集配置

# 数据集配置
dataset_type = 'ISICDataset'
data_root = '../my_custom_dataset'
crop_size = (512, 512)

# ISIC2017数据集元信息
metainfo = dict(
    classes=('background', 'lesion'),  # 二分类：背景和皮肤病变
    palette=[[0, 0, 0], [255, 255, 255]]  # 黑色背景，白色病变
)

backend_args = None

# 训练数据管道 (针对医学图像优化)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', reduce_zero_label=False, backend_args=backend_args),  # 0/1编码，保持原始标签
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.3, direction='vertical'),  # 医学图像可垂直翻转
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
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False, backend_args=backend_args),  # 0/1编码，保持原始标签
    dict(type='PackSegInputs')
]

# 训练数据加载器
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
        metainfo=metainfo,
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

# 验证数据加载器
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
        metainfo=metainfo,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# 评估指标 (二分类医学分割专用)
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],  # 添加Dice系数
    threshold=0.5,  # 二分类阈值
    nan_to_num=0
)
test_evaluator = val_evaluator