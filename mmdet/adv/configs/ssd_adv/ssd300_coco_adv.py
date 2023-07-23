_base_ = '/remote-home/liguopeng/paper/CVPR2023/mmdetection/configs/ssd/ssd300_coco.py'

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
                iou_cond = 50,
                score_cond = 5,
                optimizer = dict(type='Adamax',lr=0.1),
                regLoss = dict(type='SmoothL1Loss',reduction='mean'),
                iouLoss = dict(type='GIoULoss'),
                clsLoss = dict(type='CrossEntropyLoss',reduction='sum'),
                advPerLoss = dict(type='SmoothL1Loss',reduction='sum')),
            attacked_model = dict(
                type="SSDAdv",
                mode="hq"))