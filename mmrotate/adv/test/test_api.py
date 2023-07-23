# Copyright (c) OpenMMLab. All rights reserved.
import sys
# 添加项目根目录到运行环境, 非常重要。
sys.path.insert(0, '/remote-home/liguopeng/paper/CVPR2023/mmrotate/')

import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import cv2
import torch.distributed as dist

from mmcv.image import tensor2imgs,imwrite
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from mmcv.parallel import MMDataParallel
from mmrotate.utils import get_root_logger

from adv.builder import build_detectors,build_attacks
from adv.utils.adv_perceptible_metric import *
from adv.utils.visualize import * 
from adv.utils.FBS import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from sklearn.manifold import TSNE
import umap
import os
from mmrotate.core import imshow_det_rbboxes

from mmcv.image import imread,imwrite,bgr2rgb

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="ticks")
import umap
def single_gpu_test_adv(model,
                    data_loader,
                    cfg,
                    out_dir=None):
    logger = get_root_logger()
    logger.warning(f"Resize can bring a huge influence for adversarial examples. We remove it for the better results during attacking and testing" )
    model.eval()
    device = next(model.parameters()).device
    # collect the detection results
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # build attack methods and attacked models
    attacked_model = build_detectors(dict(**cfg.attack.attacked_model,model = model.module))
    attack = build_attacks(dict(**cfg.attack.method,model = attacked_model))
    attack = MMDataParallel(attack,device_ids=[device.index])
    PerD_images_foreground = PerceptualDistance(device=device)
    PerD_images = PerceptualDistance(device=device)
    # attack images
    iou50s = 0
    iou75s = 0
    for i,data in enumerate(data_loader):
        org_images = data['img'].data[0].to(device=device)
        img_metas = data['img_metas'].data[0]
        # if out_dir:
        #     if img_metas[0]['ori_filename'] in (os.listdir(out_dir)):
        #         norm_config = img_metas[0]['img_norm_cfg']
        #         adv_images = imread(os.path.join(out_dir,img_metas[0]['ori_filename']))
        #         adv_images = bgr2rgb(adv_images)
        #         adv_images = img2tensor(adv_images).float()
        #         adv_images = imnormalize(adv_images,inplace=False,**norm_config)
        #         adv_images = adv_images.reshape(org_images.shape).to(device)
        #     else:
        # try:
        adv_images,iou50,iou75 = attack(data)
        iou50s += iou50
        iou75s += iou75
        # except:
        #     logger = get_root_logger()
        #     logger.warning(f"{img_metas[0]['ori_filename']} failed" )
        #     adv_images = org_images
        with torch.no_grad():       
            model.eval()
            data['img'].data[0] = adv_images
            fbs = FBS(img_metas,device,1.0)
            result = model(return_loss=False, rescale=True, img = data['img'].data, img_metas = data['img_metas'].data)
            batch_size = len(result)
            # eval the perturbation
            gt_bboxes = data["gt_bboxes"].data[0]
            foreground_clean = fbs(org_images,gt_bboxes,"foreground")
            foreground_adv = fbs(adv_images,gt_bboxes,"foreground")
            
            foreground_metric = PerD_images_foreground.cal_perceptual_distances(foreground_clean,foreground_adv,img_metas)
            images_metric = PerD_images.cal_perceptual_distances(org_images,adv_images,img_metas)
            PerD_images_foreground.update(*foreground_metric,n = batch_size)
            PerD_images.update(*images_metric,n = batch_size)
        # visualize the adversarial examples
        if out_dir:
            img_tensor = data['img'].data[0]
            imgs = tensor2imgs(img_tensor.detach(), **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                # ori_h, ori_w = img_meta['ori_shape'][:-1]
                # img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                out_file = osp.join(out_dir,img_meta['ori_filename'])
                
                imwrite(img_show,out_file,params = [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    logger = get_root_logger()
    losses = attack.module.losses / len(dataset)
    bboxes = attack.module.bboxes / len(dataset)
    labels = attack.module.labels
    plt_losses(losses,labels,cfg.work_dir)
    logger.info(f"\niou50s**************:{iou50s},\niou75s***************:{iou75s}")
    logger.info(f"total running time:{prog_bar.timer.since_start()}")
    logger.info(f"mean bboxes of every iteration:\n{bboxes}")
    logger.info(f"mean losses of every iteration:\n{attack.module.losses_log(n=len(dataset))}")
    logger.info(f"metric of the adversarial examples'foreground:\n{PerD_images_foreground.metrics}")
    logger.info(f"metric of the adversarial examples:\n{PerD_images.metrics}")
    logger.info(f"0.75 FG + 0.25 img metric of the adversarial examples' foreground:\n{0.75 * PerD_images_foreground.metrics_float + 0.25 * PerD_images.metrics_float}")
            
    return results

def multi_gpu_test_adv(model, data_loader, cfg,out_dir=None,tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    logger = get_root_logger()
    
    logger.warning(f"Resize can bring a huge influence for adversarial examples. We remove it for the better results during attacking and testing" )
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    attacked_model = build_detectors(dict(**cfg.attack.attacked_model,model = model.module))
    attack = build_attacks(dict(**cfg.attack.method,model = attacked_model))
    attack = MMDataParallel(attack,device_ids=[device.index])
    PerD_images_foreground = PerceptualDistance(device=device)
    PerD_images = PerceptualDistance(device=device)
    
    # all_imgs = []
    # current_img = []
    # labels = []
    # encoder = umap.UMAP()
    iou50s = 0
    iou75s = 0
    for i, data in enumerate(data_loader):
        org_images = data['img'].data[0].to(device=device)
        img_metas = data['img_metas'].data[0]
        # if out_dir:
            # if img_metas[0]['ori_filename'] in (os.listdir(out_dir)):
            #     norm_config = img_metas[0]['img_norm_cfg']
            #     adv_images = imread(os.path.join(out_dir,img_metas[0]['ori_filename']))
            #     adv_images = bgr2rgb(adv_images)
            #     adv_images = img2tensor(adv_images).float()
            #     adv_images = imnormalize(adv_images,inplace=False,**norm_config)
            #     adv_images = adv_images.reshape(org_images.shape).to(device)
            # else:
        # try:
        adv_images,iou50,iou75 = attack(data)
        iou50s += iou50
        iou75s += iou75
        # except:
        #     logger = get_root_logger()
        #     logger.warning(f"{img_metas[0]['ori_filename']} failed" )
        #     adv_images = org_images
            
        # try:
        #     adv_images,label = attack(data)
        # except:
        #     print(i)
        #     continue
        
        # with torch.no_grad():
        #     current_img.append(adv_images.detach().cpu().numpy())
            
        #     if i%100 ==0 and i!=0:
        #         embedding = np.concatenate(current_img,axis=0).reshape(-1,adv_images.shape[-1])
        #         current_img.clear()
        #         np.save(f"/remote-home/liguopeng/object_detection/remote_sensing/mmrotate/{cfg.attack.method.type}/test{i}.npy",embedding)
        #         print(embedding.shape)
        #         del embedding
                
        with torch.no_grad():
            model.eval()
            data['img'].data[0] = adv_images
            fbs = FBS(img_metas,device,1.0)
            result = model(return_loss=False, rescale=True, img = data['img'].data, img_metas = data['img_metas'].data)
            batch_size = len(result)
            gt_bboxes = data["gt_bboxes"].data[0]
            foreground_clean = fbs(org_images,gt_bboxes,"foreground")
            foreground_adv = fbs(adv_images,gt_bboxes,"foreground")
            foreground_metric = PerD_images_foreground.cal_perceptual_distances(foreground_clean,foreground_adv,img_metas)
            images_metric = PerD_images.cal_perceptual_distances(org_images,adv_images,img_metas)
            PerD_images_foreground.update(*foreground_metric,n = batch_size)
            PerD_images.update(*images_metric,n = batch_size)
            if out_dir:
                img_tensor = data['img'].data[0]
                imgs = tensor2imgs((img_tensor).detach(), **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)
                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    # ori_h, ori_w = img_meta['ori_shape'][:-1]
                    # img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    out_file = osp.join(out_dir,img_meta['ori_filename'])
                    
                    imwrite(img_show,out_file,params = [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))
            
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
        
    if rank == 0:
        
        logger.info(f"\niou50s**************:{iou50s},\niou75s***************:{iou75s}")
        losses = attack.module.losses / len(dataset)
        labels = attack.module.labels
        bboxes = attack.module.bboxes / len(dataset)
        
        plt_losses(losses,labels,cfg.work_dir)
        
        logger.info(f"total running time:{prog_bar.timer.since_start()}")
        logger.info(f"mean bboxes of every iteration:\n{bboxes}")
        
        logger.info(f"mean losses of every iteration:\n{attack.module.losses_log(n=len(dataset))}")
        logger.info(f"metric of the adversarial examples' foreground:\n{PerD_images_foreground.metrics}")
        logger.info(f"metric of the adversarial examples:\n{PerD_images.metrics}")
        logger.info(f"0.75 FG + 0.25 img metric of the adversarial examples' foreground:\n{0.75 * PerD_images_foreground.metrics_float + 0.25 * PerD_images.metrics_float}")
    # collect results from all ranks
    if gpu_collect: 
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


