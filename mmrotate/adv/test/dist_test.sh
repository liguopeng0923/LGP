#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 PORT=29500 \
# bash /remote-home/liguopeng/paper/CVPR2023/mmrotate/adv/test/dist_test.sh \
# /remote-home/liguopeng/paper/CVPR2023/mmrotate/adv/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_adv.py \
# /remote-home/liguopeng/paper/CVPR2023/mmrotate/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth 1 --eval mAP \
# --show-dir /remote-home/liguopeng/paper/CVPR2023/cvpr2023/oriented_rcnn/PGD_Reg_fore --fuse-conv-bn

# CUDA_VISIBLE_DEVICES=1 python3 /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/adv/test/test_adv_all.py /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/adv/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_adv_all.py /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --eval mAP --work-dir /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/oriented_rcnn/DLGP_with_predicts/workdir --show-dir /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/oriented_rcnn/DLGP_with_predicts/images --fuse-conv-bn




CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/adv/test/test_adv.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
    
