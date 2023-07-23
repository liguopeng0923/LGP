import torch

from ...builder import ATTACKED_DETECTORS

import torch
from mmdet.core import bbox2roi
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from .two_stage_adv import TwoStageDetectorAdv
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
@ATTACKED_DETECTORS.register_module()
class SparseRCNNAdv(TwoStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(SparseRCNNAdv, self).__init__(model=model,mode = "last")

    def forward(self, img, img_metas, proposals=None,rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.model.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.simple_test_rpn(x, img_metas)
        bboxes,scores = self.simple_test_roi(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bboxes,scores
    
    def simple_test_rpn(self, imgs, img_metas):
        """Forward function in testing stage."""
        rpn = self.model.rpn_head
        proposals = rpn.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        # imgs_whwh has shape (batch_size, 1, 4)
        # The shape of proposals change from (num_proposals, 4)
        # to (batch_size ,num_proposals, 4)
        proposals = proposals * imgs_whwh

        init_proposal_features = rpn.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].expand(
            num_imgs, *init_proposal_features.size())
        
        return proposals, init_proposal_features, imgs_whwh
    
    def simple_test_roi(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch,
            it is a list[tuple] that contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        roi_head = self.roi_head
        
        assert roi_head.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        object_feats = proposal_features
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(roi_head.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        for stage in range(roi_head.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']
            pred_bboxes = bbox_results['decode_bbox_pred']

        num_classes = roi_head.bbox_head[-1].num_classes
        det_bboxes = []
        det_scores = []

        if roi_head.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    roi_head.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            
            bbox_pred_per_img = pred_bboxes[img_id][torch.div(topk_indices,num_classes,rounding_mode="floor")]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            if roi_head.bbox_head[0].loss_cls.use_sigmoid:
                # Add a dummy background class to the backend when using sigmoid
                # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
                # BG cat_id: num_class
                padding = cls_score_per_img.new_zeros(cls_score_per_img.shape[0], 1)
                cls_score_per_img = torch.cat([cls_score_per_img, padding], dim=1)
            det_bboxes.append(bbox_pred_per_img)
            det_scores.append(cls_score_per_img)

        return det_bboxes,det_scores
    
    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        bbox_head = self.roi_head.bbox_head[stage]
        roi_head = self.roi_head
        num_imgs = len(img_metas)
        bbox_roi_extractor = roi_head.bbox_roi_extractor[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats)
        proposal_list = bbox_head.refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=proposal_list,
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    
    

