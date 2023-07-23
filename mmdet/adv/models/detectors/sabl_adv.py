# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS

import torch
from mmdet.core import bbox2roi
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from .two_stage_adv import TwoStageDetectorAdv

@ATTACKED_DETECTORS.register_module(name="SABLAdv")
class SABLAdv(TwoStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(SABLAdv, self).__init__(model=model,mode=mode)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        bbox_head = self.roi_head.bbox_head
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes, confidences = bbox_head.bbox_coder.decode(
                rois[:, 1:], bbox_pred, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            confidences = None
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            if self.mode != "last":
                if bboxes.shape[-1] != 4:
                    fore_scores = torch.argmax(scores[:,:-1],dim=-1)
                    nums = bboxes.shape[0]
                    bboxes = bboxes.reshape(nums,-1,4)
                    bboxes = bboxes[torch.arange(nums),fore_scores]
                return bboxes.reshape(-1,4), scores 
            else:
                det_bboxes, det_logits = self.multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img,score_factors=confidences)
                return det_bboxes, det_logits
