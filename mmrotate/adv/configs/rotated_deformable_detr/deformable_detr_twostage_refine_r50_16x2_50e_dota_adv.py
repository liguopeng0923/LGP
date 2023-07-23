# -*- coding: utf-8 -*-
_base_ = '/remote-home/liguopeng/paper/CVPR2023/mmrotate/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota.py'
dataset_type = 'DOTADatasetAdv'

angle_version = {{_base_.angle_version}}

custom_imports = dict(
    imports=["adv.datasets.dota_adv"],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0., 0., 0.],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1024, 1024),
        scale_factor = 1.0,
        flip=False,
        transforms=[
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    train=dict(
        type = dataset_type,
        pipeline=train_pipeline,filter_empty_gt=True),
    val=dict(
        type=dataset_type,pipeline=test_pipeline),
    test=dict(
        type=dataset_type,pipeline=test_pipeline))

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
                wh_scale = 3,
                iou_cond = 5,
                score_cond = 5,
                optimizer = dict(type='Adamax',lr=0.1),
                regLoss = dict(type='SmoothL1Loss',reduction='mean'),
                iouLoss = dict(type='RotatedIoULoss'),
                clsLoss = dict(type='CrossEntropyLoss',reduction='sum'),
                advPerLoss = dict(type='SmoothL1Loss',reduction='sum')),
            attacked_model = dict(
                type="RotatedDeformableDETRAdv",
                mode="hq"))