# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.runner import OPTIMIZERS
from mmdet.models.builder import LOSSES
from mmcv.utils import Registry

ADVMODELS = Registry('AdvModels', parent=MMCV_MODELS)
ATTACK_METHODS = ADVMODELS
ATTACKED_DETECTORS = ADVMODELS

def build_attacks(cfg):
    """Build backbone."""
    return ATTACK_METHODS.build(cfg)

def build_detectors(cfg):
    """Build backbone."""
    return ATTACKED_DETECTORS.build(cfg)

def build_optimizer(params, cfg):
    cfg['params'] = params
    optimizer = OPTIMIZERS.build(cfg)
    return optimizer

def build_losses(cfg):
    """Build backbone."""
    return LOSSES.build(cfg)


