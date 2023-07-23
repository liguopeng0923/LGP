# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ...builder import ATTACKED_DETECTORS

from .single_stage_adv import RotatedSingleStageDetectorAdv
from mmrotate.core import poly2obb
from mmcv.ops import min_area_polygons
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

@ATTACKED_DETECTORS.register_module(name="RetinanetAdv")
class RetinanetAdv(RotatedSingleStageDetectorAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self,
                 model,
                 mode = "last"):
        super(RetinanetAdv, self).__init__(model=model,mode=mode)
        
        