faster_rcnn r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

faster_rcnn r101
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/faster_rcnn_adv/faster_rcnn_r101_fpn_1x_coco.py mmdet/checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

faster_rcnn x101
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/faster_rcnn_adv/faster_rcnn_x101_64x4d_fpn_1x_coco.py mmdet/checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```


cascade_rcnn r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/cascade_rcnn_adv/cascade_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

deformable detr r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/deformable_detr_adv/deformable_detr_twostage_refine_r50_16x2_50e_coco_adv.py mmdet/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

reppoints r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/configs/reppoints/reppoints_moment_r50_fpn_1x_coco_adv.py mmdet/checkpoints/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```


sabl r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/sabl_adv/sabl_faster_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```


sparse_rcnn r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmdet/adv/configs/sparse_rcnn_adv/sparse_rcnn_r50_fpn_1x_coco_adv.py mmdet/checkpoints/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```


tood r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/configs/tood_adv/tood_r50_fpn_1x_coco_adv.py mmdet/checkpoints/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

vfnet r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/configs/vfnet_adv/vfnet_r50_fpn_1x_coco_adv.py mmdet/checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```


oriented_rcnn r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_adv.py mmrotate/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

gliding_vertex r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py mmrotate/checkpoints/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

redet r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/redet/redet_re50_refpn_1x_dota_le90_adv.py mmrotate/checkpoints/redet_re50_fpn_1x_dota_le90-724ab2da.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

retinanet r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90_adv.py mmrotate/checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

roi_transformer r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/roi_transformer/roi_trans_r50_fpn_1x_dota_le90_adv.py mmrotate/checkpoints/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

rotated_deformable_detr r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/rotated_deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota_adv.py mmrotate/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_dota.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

rotated_fcos r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90_adv.py mmrotate/checkpoints/rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```

s2anet r50
```
CUDA_VISIBLE_DEVICES=0 python mmdet/adv/test/test_adv.py mmrotate/adv/configs/s2anet/s2anet_r50_fpn_1x_dota_le135_adv.py mmrotate/checkpoints/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth --eval bbox --work-dir ./work-dir --show-dir ./images
```
