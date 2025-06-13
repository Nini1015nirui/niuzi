_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        dims=64,
        d_state=1,
        depths=(2, 4, 8, 4),
        drop_path_rate=0.2,
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150)
)

# Quick test configuration
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Override checkpoint and logging intervals
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50),
)

# Use dummy tiny dataset
data_root = '../dummy_ade20k_tiny'
train_dataloader = dict(
    batch_size=2,
    dataset=dict(data_root='../dummy_ade20k_tiny')
)
val_dataloader = dict(
    dataset=dict(data_root='../dummy_ade20k_tiny')
)
test_dataloader = dict(
    dataset=dict(data_root='../dummy_ade20k_tiny')
)