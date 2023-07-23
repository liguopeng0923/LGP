# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.runner import OPTIMIZERS
from mmrotate.models.builder import ROTATED_LOSSES
from mmcv.utils import Registry

ADVMODELSROTATED = Registry('AdvModelsRotated', parent=MMCV_MODELS)

ATTACK_METHODS = ADVMODELSROTATED
ATTACKED_DETECTORS = ADVMODELSROTATED

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
    return ROTATED_LOSSES.build(cfg)
