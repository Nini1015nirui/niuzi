# Custom Dataset Configuration Template
# 复制此文件并根据你的数据集修改参数

# =====================================================
# 数据集基本设置
# =====================================================
dataset_type = 'ADE20KDataset'  # 使用ADE20K数据集格式
data_root = 'my_datasets/your_dataset_name'  # 修改为你的数据集路径
crop_size = (512, 512)  # 训练时的裁剪尺寸，可根据需要调整

# =====================================================
# 数据增强和预处理管道
# =====================================================
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', reduce_zero_label=True, backend_args=backend_args),
    
    # 随机缩放 - 可调整scale参数
    dict(
        type='RandomResize',
        scale=(2048, 512),  # (max_long_edge, max_short_edge)
        ratio_range=(0.5, 2.0),  # 缩放比例范围
        keep_ratio=True),
    
    # 随机裁剪到指定尺寸
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    
    # 随机水平翻转
    dict(type='RandomFlip', prob=0.5),
    
    # 光度变换（亮度、对比度等）
    dict(type='PhotoMetricDistortion'),
    
    # 打包输入
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),  # 测试时缩放
    dict(type='LoadAnnotations', reduce_zero_label=True, backend_args=backend_args),
    dict(type='PackSegInputs')
]

# =====================================================
# 数据加载器配置
# =====================================================
train_dataloader = dict(
    batch_size=4,  # 根据GPU内存调整
    num_workers=4,  # 根据CPU核心数调整
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', 
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,  # 验证通常用batch_size=1
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline,
        backend_args=backend_args,))

test_dataloader = val_dataloader

# =====================================================
# 评估指标
# =====================================================
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# =====================================================
# 测试时增强 (可选)
# =====================================================
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(int(2048*r), int(512*r)), keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], 
            [dict(type='LoadAnnotations', backend_args=backend_args)], 
            [dict(type='PackSegInputs')]
        ])
]

# =====================================================
# 使用说明
# =====================================================
"""
使用此配置的步骤：

1. 复制此文件为你的数据集名称：
   cp custom_dataset.py my_dataset_name.py

2. 修改关键参数：
   - data_root: 数据集路径
   - crop_size: 根据你的图像尺寸调整
   - batch_size: 根据GPU内存调整
   - num_classes: 在训练配置中设置

3. 在训练配置中引用：
   _base_ = ['../datasets/my_dataset_name.py', ...]
"""