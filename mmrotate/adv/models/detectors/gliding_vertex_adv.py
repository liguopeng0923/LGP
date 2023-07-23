# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from ...builder import ATTACKED_DETECTORS
from mmdet.core import bbox2roi
from mmrotate.core import obb2xyxy
from .two_stage_adv import RotatedTwoStageDetectorAdv
from mmcv.runner import force_fp32

@ATTACKED_DETECTORS.register_module(name="GlidingVertexAdv")
class GlidingVertexAdv(RotatedTwoStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(GlidingVertexAdv, self).__init__(model = model,mode = mode)
    
    def forward(self,img, img_metas, proposals=None, rescale=False):
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        x = self.model.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = [obb2xyxy(proposal) for proposal in proposals]
        
        return self.simple_test_roi(x, proposal_list, img_metas,rescale=rescale)
    
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
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)
        bbox_results = self.roi_head._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        fix_pred = bbox_results['fix_pred'],
        ratio_pred = bbox_results['ratio_pred'],
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            fix_pred = fix_pred[0].split(num_proposals_per_img, 0)
            ratio_pred = ratio_pred[0].split(num_proposals_per_img, 0)

        else:
            bbox_pred = (None, ) * len(proposals)
            fix_pred = (None, ) * len(proposals)
            ratio_pred = (None, ) * len(proposals)

        # for RAP
        if self.mode == "rpn":
            return bbox_pred[0],torch.softmax(cls_score[0],dim=-1)[:,-1]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_scores = []
        for i in range(len(proposals)):
            det_bbox, det_score = self.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                fix_pred[i],
                ratio_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=self.roi_head.test_cfg)
            det_bboxes.append(det_bbox)
            det_scores.append(det_score)

        return det_bboxes, det_scores

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'fix_pred', 'ratio_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   fix_pred,
                   ratio_pred,
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

        rbboxes = bbox_head.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(*ratio_pred.size(), 4)
        rbboxes = rbboxes.view(*ratio_pred.size(), 5)
        try:
            rbboxes[ratio_pred > bbox_head.ratio_thr] = \
                hbb2obb(bboxes[ratio_pred > bbox_head.ratio_thr], bbox_head.version)
        except:  # noqa: E722
            pass

        if rescale and rbboxes.size(0) > 0:
            scale_factor = rbboxes.new_tensor(scale_factor)
            rbboxes[..., :4] = rbboxes[..., :4] / scale_factor
            rbboxes = rbboxes.view(rbboxes.size(0), -1)

        if cfg is None:
            return rbboxes.reshape(-1,5), scores
        else:
            if self.mode != "last":
                if rbboxes.shape[-1] != 5:
                    fore_scores = torch.argmax(scores[:,:-1],dim=-1)
                    nums = rbboxes.shape[0]
                    rbboxes_all = rbboxes.reshape(nums,-1,5)
                    det_bboxes = rbboxes_all[torch.arange(nums),fore_scores]
                    return det_bboxes.reshape(-1,5),scores
                else:
                    return rbboxes.reshape(-1,5),scores
            det_bboxes, det_logits = self.multiclass_nms_rotated(
                rbboxes.reshape(-1,5), scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_logits
