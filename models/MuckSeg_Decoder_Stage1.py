import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from timm.models.layers import trunc_normal_
from models.blocks import get_ConvNeXtBlock


class MuckSeg_Decoder_Stage1(L.LightningModule):
    """3rd and 4th decoder tier of MuckSeg, used for training stage 1.

    Args:
        kernel_sizes (list): Kernel sizes of each decoder tier. Default: [7, 7]
        depths (tuple(int)): Number of blocks in each tier. Default: [2, 2]
        dim (int): Input feature dimension. Default: 512
        mlp_ratio (int): Expansion ratio of hidden feature dimension in MLP layers in each ConvNeXt block.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        use_convnext_v2 (bool): Whether to use ConvNeXt-V2 block instead of ConvNeXt block. Default: True

    Pipeline:
        <input> B, D, H/16, W/16 -(upsample0)-> B, D, H/8, W/8
             -(cat(<side_feature3>B, D/2, H/8, W/8))-> B, 3*D/2, H/8, W/8
             -(D-ConvBlock0)-> B, D/2, H/8, W/8 -> <side_output>
             -(upsample1)-> B, D/2, H/4, W/4
             -(cat(<side_feature2>B, D/4, H/4, W/4))-> B, 3*D/4, H/4, W/4
             -(D-ConvBlock1)-> B, D/4, H/4, W/4 -> <output>
    """
    _NUM_TIERS = 2


    def __init__(self, kernel_sizes=[7, 7], depths=[2, 2], dim=512, mlp_ratio=4., drop_path_rate=0., use_convnext_v2=True):
        super().__init__()
        ConvNeXtBlock = get_ConvNeXtBlock(use_convnext_v2)
        assert dim % 4 == 0, 'dim must be a multiple of 4'
        self._check_inputs(kernel_sizes, depths)
        self.side_output_detached = False

        self.decode_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self._NUM_TIERS):
            stage_modules = [nn.Conv2d(in_channels=3 * dim // (2 ** (i + 1)), out_channels=dim // (2 ** (i + 1)), kernel_size=1)]
            for j in range(depths[i]):
                stage_modules.append(ConvNeXtBlock(
                    dim=dim // (2 ** (i + 1)), kernel_size=kernel_sizes[i], mlp_ratio=mlp_ratio, drop_path=dp_rates[cur + j]
                ))
            self.decode_stages.append(nn.Sequential(*stage_modules))
            cur += depths[i]

        self.apply(self._init_weights)

    def _check_inputs(self, *args):
        parameter_lengths = set([len(arg) for arg in args])
        if len(parameter_lengths) > 1:
            raise ValueError('kernel_sizes and depths must be length 2 sequence.')
        if parameter_lengths.pop() != self._NUM_TIERS:
            raise ValueError('kernel_sizes and depths must be length 2 sequence.')

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def detach_sideoutput(self):
        self.forward = self.forward_detached
        self.side_output_detached = True

    def forward(self, x, xi):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, xi[-1]), dim=1)
        x = self.decode_stages[0](x)
        x0 = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, xi[-2]), dim=1)
        x = self.decode_stages[1](x)

        return x, x0

    def forward_detached(self, x, xi):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, xi[-1]), dim=1)
        x = self.decode_stages[0](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, xi[-2]), dim=1)
        x = self.decode_stages[1](x)

        return x

# EOF