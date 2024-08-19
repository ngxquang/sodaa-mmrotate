# dataset settings
dataset_type = 'SODAADataset'
data_root = "/storageStudents/nguyenvd/quangnx/mmrotate/data/split_sodaa/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1200, 1200)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/train/',
        img_prefix=data_root + 'Images/train/',
        pipeline=train_pipeline,
        ori_ann_file="/storageStudents/nguyenvd/dataset/SODA-A/Annotations/train/"
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/val/',
        img_prefix=data_root + 'Images/val/',
        pipeline=test_pipeline,
        ori_ann_file="/storageStudents/nguyenvd/dataset/SODA-A/Annotations/val/"
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/test/',
        img_prefix=data_root + 'Images/test/',
        pipeline=test_pipeline,
        ori_ann_file="/storageStudents/nguyenvd/dataset/SODA-A/Annotations/test/"
    ))