# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS

import torch
from mmdet.core import bbox2roi
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from .two_stage_adv import TwoStageDetectorAdv

@ATTACKED_DETECTORS.register_module(name="CascadeRCNNAdv")
class CascadeRCNNAdv(TwoStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(CascadeRCNNAdv, self).__init__(model=model,mode=mode)

    def simple_test_roi(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        roi_head = self.roi_head
        assert roi_head.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
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
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = roi_head.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        if self.mode == "rpn":
            if bbox_pred[0].shape[-1] != 4:
                fore_scores = torch.argmax(cls_score[0][:,:-1],dim=-1)
                nums = bbox_pred[0].shape[0]
                bboxes = bbox_pred[0].reshape(nums,-1,4)
                bboxes = bboxes[torch.arange(nums),fore_scores]
                # return bboxes.reshape(-1,4), torch.softmax(cls_score[0],dim=-1)[:,-1]
                return bboxes.reshape(-1,4), torch.softmax(cls_score[0],dim=-1)
            return bbox_pred[0],proposal_list[0][:,-1]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_scores = []
        for i in range(num_imgs):
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

        return det_bboxes,det_scores
