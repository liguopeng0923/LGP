# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import ATTACKED_DETECTORS
from .roi_trans_adv import RoITransformerAdv

@ATTACKED_DETECTORS.register_module(name="ReDetAdv")
class ReDetAdv(RoITransformerAdv):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 model,
                 mode = "last"):
        super(ReDetAdv, self).__init__(model = model,mode = mode)