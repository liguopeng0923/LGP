#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 PORT=29510 bash /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/tools/dist_test.sh /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/works/latest.pth 1 --eval mAP --fuse-conv-bn
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
