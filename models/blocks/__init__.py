from .auxiliaries import LayerNorm, GRN
from .stem import StemBlock
from .convnextblock import ConvNeXtBlock, ConvNeXtV2Block
from .reverse_edge_attention import ReverseEdgeAttentionBlock


__all__ = [
    'LayerNorm',
    'GRN',
    'StemBlock',
    'ConvNeXtBlock',
    'ConvNeXtV2Block',
    'ReverseEdgeAttentionBlock',
    'get_ConvNeXtBlock',
]


def get_ConvNeXtBlock(use_convnext_v2):
    if use_convnext_v2:
        return ConvNeXtV2Block
    else:
        return ConvNeXtBlock