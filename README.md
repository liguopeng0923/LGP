
## Introduction

### Towards Generic and Controllable Attacks Against Object Detection.
Existing adversarial attacks against Object Detectors (ODs) suffer from two inherent limitations. Firstly, ODs have complicated meta-structure designs, hence most advanced attacks for ODs concentrate on attacking specific detector-intrinsic structures, which makes it hard for them to work on other detectors and motivates us to design a generic attack against ODs. Secondly, most works against ODs make Adversarial Examples (AEs) by generalizing image-level attacks from classification to detection, which brings redundant computations and perturbations in semantically meaningless areas (\eg, backgrounds) and leads to an emergency for seeking controllable attacks for ODs. To this end, we propose a generic white-box attack, LGP (local perturbations with adaptively global attacks), to blind mainstream object detectors with controllable perturbations. For a detector-agnostic attack, LGP tracks high-quality proposals and optimizes three heterogeneous losses simultaneously. In this way, we can fool the crucial components of ODs with a part of their outputs without the limitations of specific structures. Regarding controllability, we establish an object-wise constraint that exploits  foreground-background separation adaptively to induce the attachment of perturbations to foregrounds. Experimentally, the proposed LGP successfully attacked sixteen state-of-the-art object detectors on MS-COCO and DOTA datasets, with promising imperceptibility and transferability obtained.



![Overall Framework](./overall_frame.png)

![Visible Results](./Visible_Results.png)



## Installation

Please refer to [install.md](docs/install.md) for installation guide.

## Get Started

Please see [MMdet_get_started.md](mmdet/docs/en/get_started.md) for the basic usage of MMdet.

Please see [MMrotate_get_started.md](mmrotate/docs/en/get_started.md) for the basic usage of MMrotate.

## Data Preparation
Please refer to [COCO.md](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html) to prepare the COCO data.

Please refer to [Dota.md](https://github.com/open-mmlab/mmrotate/blob/1.x/tools/data/dota/README.md) to prepare the DOTA data.


### Train

More runnings can be seen in [train.md](./docs/train.md)
```   
CUDA_VISIBLE_DEVICES=2 python mmdet/adv/test/test_adv.py mmdet/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

### Test
You should first modify line 52 in [data root](./mmdet/configs/_base_/datasets/coco_detection.py) to the adversarial examples dir.

```
CUDA_VISIBLE_DEVICES=2 python mmdet/tools/test.py mmdet/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

## Acknowledgement

[MMdetection](https://github.com/open-mmlab/mmdetection), and [MMRotate](https://github.com/open-mmlab/mmrotate) are open source projects that are contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.


## Citation

If you find our work useful, please consider citing our paper:
```
@misc{2307.12342,
Author = {Guopeng Li and Yue Xu and Jian Ding and Gui-Song Xia},
Title = {Towards Generic and Controllable Attacks Against Object Detection},
url = {https://arxiv.org/abs/2307.12342}
Year = {2023},
Eprint = {arXiv:2307.12342},
}
```

## Contacts

Please feel free to contact us if you have any problems.

Email: [guopengli@whu.edu.cn](guopengli@whu.edu.cn)
