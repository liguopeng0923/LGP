# Foreground separation
import torch
import cv2
import numpy as np
import torch
import math

from mmcv.image import tensor2imgs
from .transforms import *
def l2_weight(img,bbox):
    # (x,y) 任意点坐标
    # (cx,cy) 矩形中心点坐标
    # r 矩形最大内接距离
    device = img.device
    weight = torch.ones(img.shape[-2:]).to(device=device).float()
    x = torch.cumsum(weight,dim=1)
    y = torch.cumsum(weight,dim=0)
    for single_bbox in bbox:
        cx, cy, w, h, _ = single_bbox[:5]
        r = torch.sqrt(torch.pow(w / 2,2) + torch.pow(h / 2,2))
        weight = torch.min(weight,torch.sqrt(torch.pow(x - cx,2) + torch.pow(y - cy,2)) / r)
    weight = weight.expand_as(img)
    return weight

# FBS foreground and background seperate
class FBS(torch.nn.Module):

    def __init__(self,img_metas,device = torch.cuda,fore_scale = 1.0):
        super(FBS, self).__init__()
        self.img_metas = img_metas
        self.device = device
        self.fore_scale = fore_scale
    
    def forward(self,images,bboxes,type="foreground",weight=False):
        assert type=="foreground" or type=="background"
        if type=="foreground":
            return self.foreground(images,bboxes,weight)
        else:
            return self.background(images,bboxes,weight)
    
    def foreground(self,images,bboxes,weight=False):
        images = images.clone()
        img_metas = self.img_metas
        device = self.device
        fore_scale = self.fore_scale
        masks = []
        weight_masks = []
        for img,img_meta,bbox in zip(images,img_metas,bboxes):
            if bbox==[] or bbox is None or len(bbox)==0:
                masks.append(torch.zeros_like(img))
                weight_masks.append(torch.zeros_like(img))
                continue
            assert bbox is not None and bbox.ndim == 2
            image = tensor2imgs(img.unsqueeze(0).detach(),**img_meta['img_norm_cfg'])[0]
            norm_mask = ((image.astype(int) + 1 >= 0) & (image.astype(int) <= 255))
            image = np.where(norm_mask,image+1,image)
            bbox = bbox.detach().cpu()
            bbox[:,2:4] = bbox[:,2:4] * fore_scale
            for single_bbox in bbox:
                xc, yc, w, h, ag = single_bbox[:5]
                wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
                hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
                p1 = (xc - wx - hx, yc - wy - hy)
                p2 = (xc + wx - hx, yc + wy - hy)
                p3 = (xc + wx + hx, yc + wy + hy)
                p4 = (xc - wx + hx, yc - wy + hy)
                ps = np.int0(np.array([p1, p2, p3, p4]))
                image = self.getMask(image,ps)
            background = image
            mask = img2tensor(background.sum(axis=-1) == 0).to(device=device)
            mask = (mask > 0).expand_as(img)
            masks.append(mask)
            if weight:
                weight_mask = l2_weight(img,bbox)
                weight_mask = weight_mask * mask
                weight_masks.append(weight_mask)
        images = imdenormalize(images,**img_meta['img_norm_cfg'])
        foreground = [torch.mul(images[i],masks[i]) for i in range(images.shape[0])]
        foreground = torch.cat(foreground,dim=0).reshape(images.shape)
        foreground = imnormalize(foreground,**img_meta['img_norm_cfg'])
        
        if weight:
            weight_masks = torch.cat(weight_masks,dim=0).reshape(images.shape)
            return foreground,weight_masks 
        else:
            return foreground

    def background(self,images,bboxes,weight=False):
        images = images.clone()
        img_metas = self.img_metas
        device = self.device
        fore_scale = self.fore_scale
        masks = []
        weight_masks = []
        for img,img_meta,bbox in zip(images,img_metas,bboxes):
            if bbox==[] or bbox is None or len(bbox)==0:
                masks.append(torch.ones_like(img))
                weight_masks.append(torch.ones_like(img))
                continue
            assert bbox is not None and bbox.ndim == 2
            image = tensor2imgs(img.unsqueeze(0).detach(),**img_meta['img_norm_cfg'])[0]
            norm_mask = ((image.astype(int) + 1 >= 0) & (image.astype(int) <= 255))
            image = np.where(norm_mask,image+1,image)
            bbox = bbox.detach().cpu()
            bbox[:,2:4] = bbox[:,2:4] * fore_scale
            for single_bbox in bbox:
                xc, yc, w, h, ag = single_bbox[:5]
                wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
                hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
                p1 = (xc - wx - hx, yc - wy - hy)
                p2 = (xc + wx - hx, yc + wy - hy)
                p3 = (xc + wx + hx, yc + wy + hy)
                p4 = (xc - wx + hx, yc - wy + hy)
                ps = np.int0(np.array([p1, p2, p3, p4]))
                image = self.getMask(image,ps)
            background = image
            mask = img2tensor(background.sum(axis=-1) != 0).to(device=device)
            mask = (mask > 0).expand_as(img)
            masks.append(mask)
            if weight:
                weight_mask = mask.float()
                weight_masks.append(weight_mask)
        images = imdenormalize(images,**img_meta['img_norm_cfg'])
        background = [torch.mul(images[i],masks[i]) for i in range(images.shape[0])]
        background = torch.cat(background,dim=0).reshape(images.shape)
        background = imnormalize(background,**img_meta['img_norm_cfg'])
        if weight:
            weight_masks = torch.cat(weight_masks,dim=0).reshape(images.shape)
            return background,weight_masks 
        else:
            return background
    
    def getMask(self,img,ps):
        # ps为4个bbox坐标的值
        # 创建掩膜
        mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        cv2.fillConvexPoly(mask, ps, (255, 255, 255))
        bbox = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        img = img - bbox
        return img
