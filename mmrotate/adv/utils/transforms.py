import torch
import numpy as np
import torchvision.transforms as transforms

from torch import Tensor
from typing import List
from mmdet.datasets.pipelines import to_tensor

def img2tensor(img):
    # img ndarray
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = (to_tensor(img.transpose(2, 0, 1))).contiguous()
    return img

def imdenormalize(imgs,mean,std,to_rgb=False,inplace=False):
    deNorm = deNormalize(mean = mean, std = std,inplace = inplace)
    imgs = deNorm(imgs)
    return imgs

def imnormalize(imgs,mean,std,to_rgb=False,inplace=False):
    Normalize = transforms.Normalize(mean = mean, std = std,inplace=inplace)
    imgs = Normalize(imgs)
    return imgs

class deNormalize(transforms.Normalize):
    
    def __init__(self, mean, std, inplace=False):
        super(deNormalize,self).__init__(mean,std,inplace)
        

    def forward(self, tensor: Tensor) -> Tensor:
        
        return denormalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def denormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """
    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: deNormalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                        '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor
