# version_2 
# loadimagefromnpy
# one category
# augmentations
# dataset one class
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'dataset/ODFN/version_2/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 24
### byh add
classes = ('object')
### byh end
model = dict(
    backbone=dict(
        ### byh add
        in_channels=4,
        ### byh end
        base_width=4,
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCN'),
        depth=101,
        frozen_stages=1,
        groups=32,
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_32x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='pytorch',
        type='ResNeXt'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            ## byh modified
            #loss_weight=1.0,
            loss_weight=0.0,
            ## byh end
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_dfl=dict(loss_weight=0.25, type='DistributionFocalLoss'),
        ### byh change
        # num_classes=80
        num_classes=5,
        ### byh end
        reg_max=16,
        stacked_convs=4,
        type='GFLHead'),
    ### byh modified
    data_preprocessor=dict(
        # bgr_to_rgb=True,
        # mean=[
        #     123.675,
        #     116.28,
        #     103.53,
        # ],
        # pad_size_divisor=32,
        # std=[
        #     58.395,
        #     57.12,
        #     57.375,
        # ],
        type='DetDataPreprocessor'),
    ### byh end
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='GFL')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ### byh add
        serialize_data = False,
        metainfo=dict(classes=classes),
        ### byh end
        ann_file='test/annotations/test_for_1_category_5_class.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='dataset/ODFN/version_2/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromNPY'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='dataset/ODFN/version_2/test/annotations/test_for_1_category_5_class.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromNPY'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ### byh add
        serialize_data = False,
        metainfo=dict(classes=classes),
        ### byh end
        ann_file='train/annotations/train_for_1_category_5_class.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='dataset/ODFN/version_2/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromNPY'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                scale=[
                    (
                        1333,
                        480,
                    ),
                    (
                        1333,
                        800,
                    ),
                ],
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromNPY'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        scale=[
            (
                1333,
                480,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ### byh add
        serialize_data = False,
        metainfo=dict(classes=classes),
        ### byh end
        ann_file='val/annotations/val_for_1_category_5_class.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='dataset/ODFN/version_2/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromNPY'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='dataset/ODFN/version_2/val/annotations/val_for_1_category_5_class.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
