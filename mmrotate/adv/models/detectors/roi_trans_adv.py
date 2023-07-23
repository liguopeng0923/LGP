# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS
from mmdet.core import bbox2roi
from mmrotate.core import obb2xyxy
from .two_stage_adv import RotatedTwoStageDetectorAdv

@ATTACKED_DETECTORS.register_module(name="RoiTransAdv")
class RoITransformerAdv(RotatedTwoStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(RoITransformerAdv, self).__init__(model = model,mode = mode)

    def forward(self,img, img_metas, proposals=None, rescale=False):
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        x = self.model.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = [obb2xyxy(proposal) for proposal in proposals]
        
        return self.simple_test_roi(x, proposal_list, img_metas,rescale=rescale)
    
    def simple_test_roi(self, x, proposal_list, img_metas, rescale=False):
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
        roi_head = self.roi_head
        assert roi_head.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_scores = []
        rcnn_test_cfg = roi_head.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(roi_head.num_stages):
            bbox_results = roi_head._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = roi_head.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < roi_head.num_stages - 1:
                if roi_head.bbox_head[i].custom_activation:
                    cls_score = [
                        roi_head.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    roi_head.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # for RAP
        if self.mode == "rpn":
            return bbox_pred[0],torch.softmax(cls_score[0],dim=-1)[:,-1]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_logits = []
        for i in range(num_imgs):
            det_bbox, det_logit = self.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_logits.append(det_logit)

        return det_bboxes,det_logits
