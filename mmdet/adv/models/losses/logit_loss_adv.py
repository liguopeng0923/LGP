# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import LOSSES
import torch.nn as nn
@LOSSES.register_module()
class LogitLossAdv(nn.Module):

    def __init__(self,
                 reduction='sum'):
        
        super(LogitLossAdv, self).__init__()
        self.reduction = reduction

    def forward(self,
                cls_score,
                label,
                reduction_override = None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = - cls_score.gather(1,label.reshape(cls_score.shape[0],-1)).squeeze(1)
        if reduction=="mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
