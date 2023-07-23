# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ...builder import ATTACKED_DETECTORS
from mmrotate.core import rbbox2roi
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

@ATTACKED_DETECTORS.register_module(name="twoStageRotated")
class RotatedTwoStageDetectorAdv(torch.nn.Module):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "hq"):
        super(RotatedTwoStageDetectorAdv, self).__init__()
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
                5 represent (cx, cy, w, h, a, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
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

        # # for RAP
        # if self.mode == "rpn":
        #     return bbox_pred[0],torch.softmax(cls_score[0],dim=-1)[:,-1]
        
        if self.mode == "rpn":
            return bbox_pred[0],cls_score[0]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

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
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
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
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        bbox_head = self.roi_head.bbox_head
        
        if isinstance(bbox_head,torch.nn.ModuleList):
            bbox_head = bbox_head[-1]
        # some loss (Seesaw loss..) may have custom activation
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
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            if self.mode != "last":
                if bboxes.shape[-1] != 5:
                    fore_scores = torch.argmax(scores[:,:-1],dim=-1)
                    nums = bboxes.shape[0]
                    bboxes_all = bboxes.reshape(nums,-1,5)
                    det_bboxes = bboxes_all[torch.arange(nums),fore_scores].reshape(-1,5)
                    return det_bboxes,scores
                else:
                    return bboxes,scores
            det_bboxes, det_logits = self.multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_logits

    def multiclass_nms_rotated(self,
                            multi_bboxes,
                            multi_scores,
                            score_thr,
                            nms,
                            max_num=-1,
                            score_factors=None,
                            return_inds=False):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
            multi_scores (torch.Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms (float): NMS
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
        
        # 去除背景类
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 5)
            
        scores = multi_scores[:, :-1]
        logits = multi_scores.clone()

        labels = torch.arange(num_classes, dtype=torch.long)
        labels = labels.view(1, -1).expand_as(scores)
        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        
        # remove low scoring boxes
        valid_mask = scores > score_thr
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)

        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        
        mask = torch.div(inds, num_classes,rounding_mode='floor')


        logits = logits[mask]

        if bboxes.numel() == 0:
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, labels, inds
            else:
                return dets, labels

        max_coordinate = bboxes.max()
        offsets = labels.to(bboxes) * (max_coordinate + 1)

        if bboxes.size(-1) == 5:
            bboxes_for_nms = bboxes.clone()
            bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
        else:
            bboxes_for_nms = bboxes + offsets[:, None]
        _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

        if max_num > 0:
            keep = keep[:max_num]

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        logits = logits[keep]

        labels = labels.to(device = multi_bboxes.device)
        if return_inds:
            return bboxes, logits, keep
        else:
            return bboxes, logits
