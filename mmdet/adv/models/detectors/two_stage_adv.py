# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS

import torch
from mmdet.core import bbox2roi
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms

@ATTACKED_DETECTORS.register_module(name="twoStage")
class TwoStageDetectorAdv(torch.nn.Module):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(TwoStageDetectorAdv, self).__init__()
        self.model = model
        self.backbone = model.backbone
        self.device = next(model.parameters()).device

        if model.neck is not None:
            self.neck = model.neck

        if model.rpn_head is not None:
            self.rpn_head = model.rpn_head

        if model.roi_head is not None:
            self.roi_head = model.roi_head
        
        assert mode in ["roi","hq","last","rpn"]
        self.mode = mode

    def forward(self,img, img_metas, proposals=None, rescale=False):
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        x = self.model.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        
        return self.simple_test_roi(x, proposal_list, img_metas,rescale=rescale)
        
    def simple_test_roi(self,x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.roi_head.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_logits = self.simple_test_bboxes(x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)

        return det_bboxes, det_logits

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        rois = bbox2roi(proposals)
        bbox_results = self.roi_head._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.roi_head.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
            
        if self.mode == "rpn":
            if bbox_pred[0].shape[-1] != 4:
                fore_scores = torch.argmax(cls_score[0][:,:-1],dim=-1)
                nums = bbox_pred[0].shape[0]
                bboxes = bbox_pred[0].reshape(nums,-1,4)
                bboxes = bboxes[torch.arange(nums),fore_scores]
                # return bboxes.reshape(-1,4), torch.softmax(cls_score[0],dim=-1)[:,-1]
                return bboxes.reshape(-1,4), torch.softmax(cls_score[0],dim=-1)
            return bbox_pred[0],proposals[0][:,-1]
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_scores = []
        for i in range(len(proposals)):
            det_bbox, det_score = self.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_scores.append(det_score)
        return det_bboxes, det_scores

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        bbox_head = self.roi_head.bbox_head
        if isinstance(bbox_head,torch.nn.ModuleList):
            bbox_head = bbox_head[-1]
        if bbox_head.custom_cls_channels:
            scores = bbox_head.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = bbox_head.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

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
                                                    cfg.max_per_img)
                return det_bboxes, det_logits
    
    def multiclass_nms(self,multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_cfg (dict): a dict that contains the arguments of nms operations
            max_num (int, optional): if there are more than max_num bboxes after
                NMS, only top max_num will be kept. Default to -1.
            score_factors (Tensor, optional): The factors multiplied to scores
                before applying NMS. Default to None.
            return_inds (bool, optional): Whether return the indices of kept
                bboxes. Default to False.

        Returns:
            tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
                (k), and (k). Dets are boxes with scores. Labels are 0-based.
        """
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)

        scores = multi_scores[:, :-1]
        logits = multi_scores.clone()
        
        labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
        labels = labels.view(1, -1).expand_as(scores)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            # remove low scoring boxes
            valid_mask = scores > score_thr
        # multiply score_factor after threshold to preserve more bboxes, improve
        # mAP by 1% for YOLOv3
        if score_factors is not None:
            # expand the shape to match original shape of score
            logit_factors = score_factors.view(-1,1).expand(
                multi_scores.size(0), num_classes+1
            )
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors
            logits = logits * logit_factors

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
            mask = torch.div(inds, num_classes,rounding_mode='floor')
            logits = logits[mask]
        else:
            # TensorRT NMS plugin has invalid output filled with -1
            # add dummy data to make detection output correct.
            bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
            scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
            labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
            logits = torch.cat([logits, logits.new_zeros(1)], dim=0)
        
        if bboxes.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                'as it has not been executed this time')
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, logits, inds
            else:
                return dets, logits
            
        
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        if return_inds:
            return dets[:,:-1], logits[keep], inds[keep]
        else:
            return dets[:,:-1], logits[keep]

