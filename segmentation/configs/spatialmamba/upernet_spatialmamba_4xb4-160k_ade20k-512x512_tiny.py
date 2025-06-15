_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/isic2017.py',  # 🔧 使用ISIC2017数据集配置
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# 数据预处理器配置
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),  # 固定输入尺寸
    size_divisor=None  # 不使用divisor
)

# 模型配置 - Spatial-Mamba Tiny + ISIC2017二分类
model = dict(
    pretrained=None,  # 禁用ResNet50预训练权重
    data_preprocessor=data_preprocessor,  # 添加数据预处理器
    backbone=dict(
        _delete_=True,  # 删除基础配置的backbone
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        dims=64,        # Tiny: 64维
        d_state=1,      # Mamba状态维度
        depths=(2, 4, 8, 4),  # 各层深度
        drop_path_rate=0.1,   # Drop path比率
        recompute=True,       # 梯度检查点减少显存
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained_weights/upernet_spatialmamba_4xb4-160k_ade20k-512x512_tiny_iter_144000.pth',
            prefix='backbone.',      # 🔧 仅加载backbone权重
            strict=False,           # 允许部分权重不匹配
            map_location='cpu',     # 避免内存问题
        ),
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],  # 对应Tiny backbone输出
        num_classes=1,                    # 🔥 二分类：单通道输出
        threshold=0.5,                    # 🔧 显式二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 分类头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,             # 🔧 二分类使用sigmoid
            class_weight=[2.0],           # 病变类别权重
            avg_non_ignore=True
        )
    ),
    auxiliary_head=dict(
        in_channels=256,                  # 来自backbone第3层
        num_classes=1,                    # 🔥 与decode_head保持一致
        threshold=0.5,                    # 🔧 显式二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 辅助头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,             # 🔧 二分类使用sigmoid
            class_weight=[2.0],           # 病变类别权重
            avg_non_ignore=True
        )
    )
)

# 训练配置优化 (针对ISIC2017)
max_iters = 80000
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=2000  # 每2000次迭代验证
)

# 学习率调度 (医学图像优化)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False)
]

# 优化器配置 - 使用SGD避免配置冲突
optimizer = dict(
    type='SGD', 
    lr=0.003,          # 适合医学图像的学习率
    momentum=0.9, 
    weight_decay=0.0005
)
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=optimizer, 
    clip_grad=None
)

# 日志和检查点配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,     # 每2000次迭代保存
        max_keep_ckpts=3
    ),
    logger=dict(
        type='LoggerHook',
        interval=100,      # 每100次迭代打印日志
        log_metric_by_epoch=False
    )
)

# 工作目录
work_dir = './work_dirs/upernet_spatialmamba_tiny_isic2017'
