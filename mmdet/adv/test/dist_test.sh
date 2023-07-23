#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=1 PORT=29522 bash /remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/dist_test.sh /remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py /remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth 1 --eval bbox --work-dir /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/faster_rcnn/10_iter/workdir  --show-dir /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/faster_rcnn/10_iter/images --fuse-conv-bn 

# "/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py",
#                 "/remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
#                 "--eval","bbox",
#             "--work-dir","/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/results/workdir","--show-dir","/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/results/advimages",

# "--nnodes","1",
# "--node_rank","0",
# "--master_addr","127.0.0.1",
# "--nproc_per_node","2",
# "--master_port","29500",
# "/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/test_adv.py",
# "/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/configs/faster_rcnn_adv/faster_rcnn_r50_fpn_1x_coco_adv.py",
# "/remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
# "--launcher","pytorch","--eval","bbox",
# "--work-dir","/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/results/workdir","--show-dir","/remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/results/advimages"

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    /remote-home/liguopeng/object_detection/object_detection/mmdetection/adv/test/test_adv.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
    
