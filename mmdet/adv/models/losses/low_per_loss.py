# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import LOSSES
from mmdet.models.losses import SmoothL1Loss
import torch.nn as nn
from adv.utils.DWT import *

@LOSSES.register_module()
class LowPerLoss(nn.Module):

    def __init__(self,
                 reduction='sum',
                 wavename='haar'):
        
        super(LowPerLoss, self).__init__()
        self.reduction = reduction
        self.DWT = DWT_2D_tiny(wavename= wavename)
        self.IDWT = IDWT_2D_tiny(wavename= wavename)
        self.smoothl1loss = SmoothL1Loss(reduction=reduction)

    def forward(self,
                org,
                adv,
                weight = 1,
                reduction_override = None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        clean_ll = self.DWT(org)
        clean_ll = self.IDWT(clean_ll)
        
        adv_ll = self.DWT(adv)
        adv_ll = self.IDWT(adv_ll)
        
        loss = self.smoothl1loss(clean_ll,adv_ll,reduction_override = reduction,weight = weight)
        return loss
