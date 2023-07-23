# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import sys
# 添加项目根目录到运行环境, 非常重要。
sys.path.insert(0, '/remote-home/liguopeng/object_detection/object_detection/mmdetection')
# "/remote-home/liguopeng/adversarial_attack/detection/DAG/output/images/0_adv.jpg","/remote-home/liguopeng/object_detection/object_detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py", "/remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", "--out-file", "/remote-home/liguopeng/object_detection/object_detection/mmdetection/demo/res.png"
# python /remote-home/liguopeng/object_detection/object_detection/mmdetection/demo/image_demo.py /remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/faster_rcnn/DLGP_roi/1.5/images/000000006818.png /remote-home/liguopeng/object_detection/object_detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py /remote-home/liguopeng/object_detection/object_detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --out-file /remote-home/liguopeng/object_detection/object_detection/mmdetection/demo/res.png
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results

    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)