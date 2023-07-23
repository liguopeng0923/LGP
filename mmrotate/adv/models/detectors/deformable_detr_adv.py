# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmcv.runner import force_fp32
from ...builder import ATTACKED_DETECTORS
from .single_stage_adv import RotatedSingleStageDetectorAdv
from mmrotate.core.bbox import hbb2obb
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
@ATTACKED_DETECTORS.register_module(name="RotatedDeformableDETRAdv")
class RotatedDeformableDETRAdv(RotatedSingleStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(RotatedDeformableDETRAdv, self).__init__(model=model,mode=mode)
        
    def forward(self,img, img_metas, proposals=None, rescale=False):
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        for i,img_meta in enumerate(img_metas):
            img_meta['batch_input_shape'] = tuple(img[i].size()[-2:])
        feat = self.model.extract_feat(img)
        outs = self.bbox_head(feat,img_metas)
        bboxes, scores= self.get_bboxes(
            *outs, img_metas, rescale=rescale)
        
        return bboxes,scores

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        bboxes_list = []
        scores_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            bbox, score = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            bboxes_list.append(bbox)
            scores_list.append(score)
        return bboxes_list,scores_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        bbox_head = self.bbox_head
        assert len(cls_score) == len(bbox_pred)
        max_per_img = bbox_head.test_cfg.get('max_per_img', bbox_head.num_query)
        # exclude background
        if bbox_head.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            logits = cls_score.clone()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % bbox_head.num_classes
            bbox_index = indexes // bbox_head.num_classes
            bbox_pred = bbox_pred[bbox_index]
            logits = logits[bbox_index]
            
        else:
            logits = cls_score.clone()
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            logits = logits[bbox_index]
            det_labels = det_labels[bbox_index]

        bbox_pred[..., :4] = bbox_pred[..., :4] * img_shape[1]
        if rescale:
            bbox_pred[:, :4] /= bbox_pred[:, :4].new_tensor(scale_factor)
        
        padding = logits.new_zeros(logits.shape[0], 1)
        logits = torch.cat([logits, padding], dim=1)
        return bbox_pred, logits