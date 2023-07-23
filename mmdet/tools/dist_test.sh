#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 PORT=29503 bash /remote-home/liguopeng/object_detection/object_detection/mmdetection/tools/dist_test.sh \
#  /remote-home/liguopeng/object_detection/object_detection/mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py\
#     /remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth\
#     1   --eval bbox --work-dir /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/mmdetection/sabl_r50/workdir

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
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
