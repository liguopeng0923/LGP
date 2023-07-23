# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import ATTACKED_DETECTORS

from .single_stage_adv import SingleStageDetectorAdv
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
import torch

@ATTACKED_DETECTORS.register_module()
class SSDAdv(SingleStageDetectorAdv):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""


    def __init__(self,
                 model,
                 mode = "hq"):
        super(SSDAdv, self).__init__(model=model,mode = mode) 
    
    