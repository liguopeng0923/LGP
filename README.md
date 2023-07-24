
## Introduction

### Towards Generic and Controllable Attacks Against Object Detection.
Existing adversarial attacks against Object Detectors (ODs) suffer from two inherent limitations. Firstly, ODs have complicated meta-structure designs, hence most advanced attacks for ODs concentrate on attacking specific detector-intrinsic structures, which makes it hard for them to work on other detectors and motivates us to design a generic attack against ODs. Secondly, most works against ODs make Adversarial Examples (AEs) by generalizing image-level attacks from classification to detection, which brings redundant computations and perturbations in semantically meaningless areas (\eg, backgrounds) and leads to an emergency for seeking controllable attacks for ODs. To this end, we propose a generic white-box attack, LGP (local perturbations with adaptively global attacks), to blind mainstream object detectors with controllable perturbations. For a detector-agnostic attack, LGP tracks high-quality proposals and optimizes three heterogeneous losses simultaneously. In this way, we can fool the crucial components of ODs with a part of their outputs without the limitations of specific structures. Regarding controllability, we establish an object-wise constraint that exploits  foreground-background separation adaptively to induce the attachment of perturbations to foregrounds. Experimentally, the proposed LGP successfully attacked sixteen state-of-the-art object detectors on MS-COCO and DOTA datasets, with promising imperceptibility and transferability obtained.



![overall framework](./docs/overall_frame.png)

![overall framework](./docs/More_results.png)



## Installation

Please refer to [install.md](docs/install.md) for installation guide.

## Get Started

Please see [MMdet_get_started.md](mmdet/docs/en/get_started.md) for the basic usage of MMdet.

Please see [MMrotate_get_started.md](mmrotate/docs/en/get_started.md) for the basic usage of MMrotate.

## Data Preparation
Please refer to [COCO.md](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html) to prepare the COCO data.

Please refer to [Dota.md](https://github.com/open-mmlab/mmrotate/blob/1.x/tools/data/dota/README.md) to prepare the DOTA data.


### Train
```   
CUDA_VISIBLE_DEVICES=2 python mmdet/adv/test/test_adv.py mmdet/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

### Test
```
CUDA_VISIBLE_DEVICES=2 python mmdet/tools/tes.py mmdet/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

## Acknowledgement

MMdetection, and MMRotate are open source projects that are contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}


@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}


@misc{kastryulin2022piq,
  title = {PyTorch Image Quality: Metrics for Image Quality Assessment},
  url = {https://arxiv.org/abs/2208.14818},
  author = {Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V.},
  doi = {10.48550/ARXIV.2208.14818},
  publisher = {arXiv},
  year = {2022}
}

```

## Citation

If you use LGP in your projects or papers, please, cite it as follows.
```

```