# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import ATTACKED_DETECTORS

from .single_stage_adv import SingleStageDetectorAdv


@ATTACKED_DETECTORS.register_module()
class VFNetAdv(SingleStageDetectorAdv):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 model,
                 mode = "last"):
        super(VFNetAdv, self).__init__(model=model,mode = mode)
        
