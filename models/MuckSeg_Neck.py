import torch.nn as nn
import lightning as L
from models.blocks import get_ConvNeXtBlock


class MuckSeg_Neck(L.LightningModule):
    def __init__(self, kernel_size=3, dim=512, depth=2, mlp_ratio=4., drop_path=0., use_convnext_v2=False):
        super().__init__()
        ConvNeXtBlock = get_ConvNeXtBlock(use_convnext_v2)
        self.layers = nn.Sequential(*[ConvNeXtBlock(
            dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio, drop_path=drop_path
        ) for j in range(depth)])

    def forward(self, x):
        return self.layers(x)

# EOF