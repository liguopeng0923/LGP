# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS

from .single_stage_adv import RotatedSingleStageDetectorAdv
from mmrotate.core import poly2obb
from mmcv.ops import min_area_polygons
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

@ATTACKED_DETECTORS.register_module(name="S2ANetAdv")
class S2ANetAdv(RotatedSingleStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self,
                 model,
                 mode = "last"):
        super(S2ANetAdv, self).__init__(model=model,mode=mode)

    def forward(self, img, img_metas,proposals=None, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.model.extract_feat(img)
        outs = self.model.fam_head(x)
        rois = self.model.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.model.align_conv(x, rois)
        outs = self.model.odm_head(align_feat)

        bbox_inputs = outs + (img_metas, self.model.test_cfg, rescale)
        bboxes, scores = self.get_bboxes(*bbox_inputs, rois=rois)
        
        return bboxes,scores
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None):
        """Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rois (list[list[Tensor]]): input rbboxes of each level of
            each image. rois output by former stages and are to be refined
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (xc, yc, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

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
            bboxes,scores = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                rois[img_id], img_shape,
                                                scale_factor, cfg, rescale)
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
        odm_head = self.model.odm_head
        cfg = odm_head.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, odm_head.cls_out_channels)
            if odm_head.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if odm_head.use_sigmoid_cls:
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
            bboxes = odm_head.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if odm_head.use_sigmoid_cls:
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

