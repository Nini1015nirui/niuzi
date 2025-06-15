_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/isic2017.py',  # 🔧 使用ISIC2017数据集配置
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# 模型配置 - Spatial-Mamba Base + ISIC2017二分类
model = dict(
    pretrained=None,  # 禁用ResNet50预训练权重
    backbone=dict(
        _delete_=True,  # 删除基础配置的backbone
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        dims=128,       # Base: 128维
        d_state=1,      # Mamba状态维度
        depths=(2, 4, 21, 5),  # 各层深度
        drop_path_rate=0.5,   # Drop path比率
        recompute=True,       # 梯度检查点减少显存
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained_weights/upernet_spatialmamba_4xb4-160k_ade20k-512x512_base_iter_144000.pth',
            prefix='backbone.',      # 🔧 仅加载backbone权重
            strict=False,           # 允许部分权重不匹配
            map_location='cpu',     # 避免内存问题
        ),
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],  # 对应Base backbone输出
        num_classes=1,                      # 🔥 二分类：单通道输出
        threshold=0.5,                      # 🔧 显式二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 分类头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,               # 🔧 二分类使用sigmoid
            class_weight=[2.0],             # 病变类别权重
            avg_non_ignore=True
        )
    ),
    auxiliary_head=dict(
        in_channels=512,                    # 来自backbone第3层
        num_classes=1,                      # 🔥 与decode_head保持一致
        threshold=0.5,                      # 🔧 显式二分类阈值
        init_cfg=dict(type='Normal', std=0.01),  # 🆕 辅助头随机初始化
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,               # 🔧 二分类使用sigmoid
            class_weight=[2.0],             # 病变类别权重
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

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00003,        # 较小学习率适合医学图像
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
work_dir = './work_dirs/upernet_spatialmamba_base_isic2017'


