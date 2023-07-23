# Copyright (c) OpenMMLab. All rights reserved.
from .two_stage_adv import RotatedTwoStageDetectorAdv
from .single_stage_adv import RotatedSingleStageDetectorAdv
from .deformable_detr_adv import RotatedDeformableDETRAdv
from .roi_trans_adv import RoITransformerAdv
from .redet_adv import ReDetAdv
from .rotated_fcos_adv import RotatedFcosAdv
from .s2anet_adv import S2ANetAdv
from .gliding_vertex_adv import GlidingVertexAdv
from .retinanet_adv import RetinanetAdv
__all__ = [
    'RotatedTwoStageDetectorAdv','RotatedSingleStageDetectorAdv','RotatedDeformableDETRAdv','ReDetAdv','RoITransformerAdv','RotatedFcosAdv','S2ANetAdv','GlidingVertexAdv','RetinanetAdv'
]
