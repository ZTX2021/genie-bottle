from typing import List, Tuple
import torch.nn as nn

from .attention import SpaceTimeAttention
from .attention import SpatialAttention
from .attention import TemporalAttention

from genie.utils import Blueprint, default, exists
from .image import BlurPooling2d
from .image import SpaceDownsample
from .image import ImageResidualBlock

from .norm import AdaptiveGroupNorm

from .video import CausalConv3d
from .video import VideoResidualBlock
from .video import CausalConvTranspose3d
from .video import DepthToSpaceTimeUpsample
from .video import DepthToSpaceUpsample
from .video import DepthToTimeUpsample
from .video import SpaceTimeDownsample

def get_module(name : str) -> nn.Module:
    match name:
        # * Attention modules
        case 'space_attn':
            return SpatialAttention
        case 'time_attn':
            return TemporalAttention
        case 'space-time_attn':
            return SpaceTimeAttention
        # * Image modules
        case 'blur_pool':
            return BlurPooling2d
        case 'space_downsample':
            return SpaceDownsample
        case 'image-residual':
            return ImageResidualBlock
        # * Video modules
        case 'video-residual':
            return VideoResidualBlock
        case 'causal-conv3d':
            return CausalConv3d
        case 'causal-conv3d-transpose':
            return CausalConvTranspose3d
        case 'depth2space_upsample':
            return DepthToSpaceUpsample
        case 'depth2time_upsample':
            return DepthToTimeUpsample
        case 'depth2spacetime_upsample':
            return DepthToSpaceTimeUpsample
        case 'spacetime_downsample':
            return SpaceTimeDownsample
        # * Basic convolution modules
        case 'conv':
            return nn.Conv2d
        case 'conv3d':
            return nn.Conv3d
        case 'conv_transpose':
            return nn.ConvTranspose2d
        case 'conv3d_transpose':
            return nn.ConvTranspose3d
        # * Norm modules
        case 'group_norm':
            return nn.GroupNorm
        case 'adaptive_group_norm':
            return AdaptiveGroupNorm
        # * Activation modules
        case 'gelu':
            return nn.GELU
        case 'relu':
            return nn.ReLU
        case 'leaky_relu':
            return nn.LeakyReLU
        case 'silu':
            return nn.SiLU
        case _:
            raise ValueError(f'Unknown module name: {name}')
        
def parse_blueprint(
    blueprint : Blueprint,
) -> Tuple[nn.ModuleList, List[bool]]:
    # Parse the blueprint
    layers = []
    ext_kw = []
    
    for desc in blueprint:            
        if isinstance(desc, str): desc = (desc, {})
        
        name, kwargs = default(desc, (None, {}))
        ext_kw.extend(
            [kwargs.pop('has_ext', False)] * kwargs.get('n_rep', 1)
        )
        layers.extend(
            [
                get_module(name)(**kwargs)
                for _ in range(kwargs.pop('n_rep', 1))
                if exists(name) and exists(kwargs)
            ]
        )
        
    return nn.ModuleList(layers), ext_kw