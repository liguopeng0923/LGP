_base_ = '/remote-home/liguopeng/paper/CVPR2023/mmdetection/configs/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# model = dict(
#     test_cfg=dict(
#         rpn=dict(
#             nms=dict(type='nms', iou_threshold=0.9))
#     ))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        scale_factor = 1.0,
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        pipeline=train_pipeline),
    test=dict(
        pipeline=test_pipeline))

attack = dict(
            method = dict(
                type="LGP",
                num_iteration=50,
                alpha = 1,
                beta= 1,
                gamma= 1,
                miu = 0.1,
                single_stage = True,
                adv_labels = -1,
                fore_scale=1.5,
                wh_scale = 0.1,
                iou_cond = 5,
                score_cond = 5,
                optimizer = dict(type='Adamax',lr=0.1),
                regLoss = dict(type='SmoothL1Loss',reduction='mean'),
                iouLoss = dict(type='GIoULoss'),
                clsLoss = dict(type='CrossEntropyLoss',reduction='sum'),
                advPerLoss = dict(type='SmoothL1Loss',reduction='sum')),
            attacked_model = dict(
                type="ReppointsAdv",
                mode="hq"))