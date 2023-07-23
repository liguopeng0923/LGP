# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys
# 添加项目根目录到运行环境, 非常重要。
sys.path.insert(0, '/remote-home/liguopeng/object_detection/remote_sensing/mmrotate')
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401
"""
CUDA_VISIBLE_DEVICES=0 python /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/demo/image_demo.py \
/remote-home/liguopeng/object_detection/remote_sensing/mmrotate/data/split_ss_dota/val/images/P0007__1024__0___824.png \
/remote-home/liguopeng/object_detection/remote_sensing/mmrotate/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py \
/remote-home/liguopeng/object_detection/remote_sensing/mmrotate/checkpoints/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth \
--out-file /remote-home/liguopeng/object_detection/remote_sensing/mmrotate/demo/RR.png
"""
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
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)
    # show the results
    model.show_result(
        args.img, result, bbox_color="green",text_color= "green",score_thr=args.score_thr, out_file=args.out_file)



if __name__ == '__main__':
    args = parse_args()
    main(args)