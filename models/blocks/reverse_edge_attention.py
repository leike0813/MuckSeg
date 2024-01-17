import torch
import torch.nn as nn


class ReverseEdgeAttentionBlock(nn.Module):
    """This module is derived from 'OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers. https://arxiv.org/abs/2207.02255'

        All copyrights reserved by the original author.

    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.layer = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, region_out, boundary_feature):
        avg = torch.mean(region_out, dim=1, keepdim=True)
        max, _ = torch.max(region_out, dim=1, keepdim=True)
        attn = 1 - self.layer(torch.cat((avg, max), dim=1)).sigmoid()
        return boundary_feature * attn
