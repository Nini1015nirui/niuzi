_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/ade20k.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        _delete_=True,  # 删除基础配置的backbone
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        dims=64,
        d_state=1,
        depths=(2, 4, 8, 4),
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../detection/work_dirs/detection_tiny_test/epoch_5.pth',
            prefix='backbone.',  # 只加载backbone部分
        ),
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512], 
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=256, 
        num_classes=150
    )
)

# Override dataset configuration for tiny dataset
dataset_type = 'ADE20KDataset'
data_root = '../dummy_ade20k_tiny'  # Corrected path

# Update train and val dataloaders
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
    ))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
    ))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
    ))

# Shorter training for tiny dataset
train_cfg = dict(max_iters=500, val_interval=100)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=500,
        by_epoch=False,
    )
]


