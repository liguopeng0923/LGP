from mmrotate.models.builder import ROTATED_BACKBONES
from mmrotate.models.backbones.re_resnet import ReResNet

@ROTATED_BACKBONES.register_module()
class ReResNetAdv(ReResNet):

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            if not self.deep_stem:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            # m.eval()
            for param in m.parameters():
                param.requires_grad = False
