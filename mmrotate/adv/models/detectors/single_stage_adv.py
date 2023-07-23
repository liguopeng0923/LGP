# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

@ATTACKED_DETECTORS.register_module(name="singleStageRotated")
class RotatedSingleStageDetectorAdv(torch.nn.Module):
    """Base class for rotated one-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(RotatedSingleStageDetectorAdv, self).__init__()
        self.model = model
        self.backbone = model.backbone
        self.device = next(model.parameters()).device
        
        if model.neck is not None:
            self.neck = model.neck
        if hasattr(model,'bbox_head') and model.bbox_head is not None:
            self.bbox_head = model.bbox_head

        assert mode in ["roi","hq","last"]
        self.mode = mode

    def forward(self,img, img_metas, proposals=None, rescale=False):
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        x = self.model.extract_feat(img)
        outs = self.bbox_head(x)
        bboxes, scores= self.get_bboxes(
            *outs, img_metas, rescale=rescale)
        
        return bboxes,scores

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        bbox_head = self.bbox_head
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = bbox_head.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = []
        scores_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id] for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id] for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                bboxes,scores = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                bboxes,scores = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
        return bboxes_list,scores_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        bbox_head = self.bbox_head
        cfg = bbox_head.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, bbox_head.cls_out_channels)
            if bbox_head.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if bbox_head.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = bbox_head.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if bbox_head.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            if self.mode != "last":
                if mlvl_bboxes.shape[-1] != 5:
                    fore_scores = torch.argmax(mlvl_scores[:,:-1],dim=-1)
                    nums = mlvl_bboxes.shape[0]
                    bboxes_all = mlvl_bboxes.reshape(nums,-1,5)
                    det_bboxes = bboxes_all[torch.arange(nums),fore_scores].reshape(-1,5)
                    return det_bboxes,mlvl_scores
                else:
                    return mlvl_bboxes,mlvl_scores
            else:
                det_bboxes, det_scores = self.multiclass_nms_rotated(
                    mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                    cfg.max_per_img)
                return det_bboxes, det_scores
        else:
            return mlvl_bboxes, mlvl_scores

    def multiclass_nms_rotated(self,multi_bboxes,
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
            nms (float): Config of NMS.
            max_num (int, optional): if there are more than max_num bboxes after
                NMS, only top max_num will be kept. Default to -1.
            score_factors (Tensor, optional): The factors multiplied to scores
                before applying NMS. Default to None.
            return_inds (bool, optional): Whether return the indices of kept
                bboxes. Default to False.

        Returns:
            tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
        """
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
            logit_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes + 1)
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors
            logits = logits * logit_factors

        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        mask = torch.div(inds, num_classes,rounding_mode='floor')
        logits = logits[mask]
        
        
        if bboxes.numel() == 0:
            if return_inds:
                return bboxes, logits, inds
            else:
                return bboxes, logits

        # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
        # should be calculated by polygon coordinates.
        # But the conversion from rbbox to polygon will slow down the speed.
        # So we use max(x,y) + max(w,h) as max coordinate
        # which is larger than polygon max coordinate
        # max(x1, y1, x2, y2,x3, y3, x4, y4)
        max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
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
        
        if return_inds:
            return bboxes, logits, keep
        else:
            return bboxes, logits
    
    