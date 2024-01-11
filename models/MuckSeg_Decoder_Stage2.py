import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from timm.models.layers import trunc_normal_
from models.blocks import ReverseEdgeAttentionBlock, get_ConvNeXtBlock


class MuckSeg_Decoder_Stage2(L.LightningModule):
    """
    Args:
        region_kernel_sizes (tuple(int)): Kernel sizes of each decoder tier in region pipeline. Default: [7, 7]
        region_depths (tuple(int)): Number of blocks in each tier in region pipeline. Default: [2, 2]
        boundary_kernel_sizes (tuple(int)): Kernel sizes of each decoder tier in boundary pipeline. Default: [7, 7]
        boundary_depths (tuple(int)): Number of blocks in each tier in boundary pipeline. Default: [2, 2]
        dim (int): Input feature dimension. Default: 128
        mlp_ratio (int): Expansion ratio of hidden feature dimension in MLP layers in each ConvNeXt block.
        drop_path_rate (2-sequence of float): Stochastic depth rate (from deepest tier to shallowest tier). Default: [0., 0.]
        rea_kernel_size (int): Kernel size of REA block. Default: 7
        use_convnext_v2 (bool): Whether to use ConvNeXt-V2 block instead of ConvNeXt block. Default: True

    Pipeline:
        <input> {B, 4*D, H/16, W/16 -(upsample0)-> B, 4*D, H/8, W/8
             -(cat(<side_feature3>B, 2*D, H/8, W/8))-> B, 6*D, H/8, W/8
             -(D-ConvBlock0)-> B, 2*D, H/8, W/8 -> <side_output>
             -(upsample1)-> B, 2*D, H/4, W/4
             -(cat(<side_feature2>B, D, H/4, W/4))-> B, 3*D, H/4, W/4
             -(D-ConvBlock1)-> B, D, H/4, W/4 -> <output> | STAGE1_DECODER}
             ___________________________________________________________________---
             <output> -(upsample0)-> B, D, H/2, W/2 -(cat(<side_feature1>B, D/2, H/2, W/2))-> B, 3*D/2, H/2, W/2 <BASE>
                 Branch_Region: <BASE> -(D-ConvBlock0)-> B, D/2, H/2, W/2 <Region-side_output>
                 -(upsample1)-> B, D/2, H, W -(cat(<side_feature0>B, D/4, H, W))-> B, 3*D/4, H, W
                 -(D-ConvBlock1)-> B, D/4, H, W <Region-output>

                Branch_Boundary:  <Region-side_output> -(REA0)-> B, 1, H/2, W/2 -(fuse<BASE>)-> B, 3*D/2, H/2, W/2
                -(D-ConvBlock0)-> B, D/2, H/2, W/2 <Boundary-side_output>
                 -(upsample1)-> B, D/2, H, W -(cat(<side_feature0>B, D/4, H, W))-> B, 3*D/4, H, W <CKPT>
                 <Region-output> -(REA1)-> B, 1, H, W -(fuse<CKPT>)-> B, 3*D/4, H, W
                 -(D-ConvBlock1)-> B, D/4, H, W <Boundary-output>
    """
    _NUM_TIERS = 2


    def __init__(self, region_kernel_sizes=[7, 7], region_depths=[2, 2],
                 boundary_kernel_sizes=[7, 7], boundary_depths=[2, 2],
                 dim=128, mlp_ratio=4., drop_path_rate=[0., 0.], rea_kernel_size=7, use_convnext_v2=True):
        super().__init__()
        ConvNeXtBlock = get_ConvNeXtBlock(use_convnext_v2)
        assert dim % 4 == 0, 'dim must be a multiple of 4'
        self._check_inputs(region_kernel_sizes, region_depths, boundary_kernel_sizes, boundary_depths)
        self.side_output_detached = False

        self.REA0 = ReverseEdgeAttentionBlock(kernel_size=rea_kernel_size)
        self.REA1 = ReverseEdgeAttentionBlock(kernel_size=rea_kernel_size)

        self.region_decode_stages = nn.ModuleList()
        region_dp_rates = [x.item() for x in torch.linspace(drop_path_rate[0], drop_path_rate[1], sum(region_depths))]
        cur = 0
        for i in range(self._NUM_TIERS):
            region_stage_modules = [nn.Conv2d(
                in_channels=3 * dim // (2 ** (i + 1)),
                out_channels=dim // (2 ** (i + 1)),
                kernel_size=1
            )]
            for j in range(region_depths[i]):
                region_stage_modules.append(ConvNeXtBlock(
                    dim=dim // (2 ** (i + 1)),
                    kernel_size=region_kernel_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop_path=region_dp_rates[cur + j]
                ))

            self.region_decode_stages.append(nn.Sequential(*region_stage_modules))
            cur += region_depths[i]

        self.boundary_decode_stages = nn.ModuleList()
        boundary_dp_rates = [x.item() for x in torch.linspace(drop_path_rate[0], drop_path_rate[1], sum(boundary_depths))]
        cur = 0
        for i in range(self._NUM_TIERS):
            boundary_stage_modules = [nn.Conv2d(
                in_channels=3 * dim // (2 ** (i + 1)),
                out_channels=dim // (2 ** (i + 1)),
                kernel_size=1
            )]
            for j in range(boundary_depths[i]):
                boundary_stage_modules.append(ConvNeXtBlock(
                    dim=dim // (2 ** (i + 1)),
                    kernel_size=boundary_kernel_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop_path=boundary_dp_rates[cur + j]
                ))
            self.boundary_decode_stages.append(nn.Sequential(*boundary_stage_modules))
            cur += boundary_depths[i]

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
        x = torch.cat((x, xi[-3]), dim=1)

        xr = self.region_decode_stages[0](x)
        xr0 = xr
        xb = self.REA0(xr, x)
        xb = self.boundary_decode_stages[0](xb)
        xb0 = xb

        xr = F.interpolate(xr, scale_factor=2, mode='bilinear')
        xr = torch.cat((xr, xi[-4]), dim=1)
        xr = self.region_decode_stages[1](xr)

        xb = F.interpolate(xb, scale_factor=2, mode='bilinear')
        xb = torch.cat((xb, xi[-4]), dim=1)
        xb = self.REA1(xr, xb)
        xb = self.boundary_decode_stages[1](xb)

        return xb, xr, xb0, xr0

    def forward_detached(self, x, xi):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, xi[-3]), dim=1)

        xr = self.region_decode_stages[0](x)
        xb = self.REA0(xr, x)
        xb = self.boundary_decode_stages[0](xb)

        xr = F.interpolate(xr, scale_factor=2, mode='bilinear')
        xr = torch.cat((xr, xi[-4]), dim=1)
        xr = self.region_decode_stages[1](xr)

        xb = F.interpolate(xb, scale_factor=2, mode='bilinear')
        xb = torch.cat((xb, xi[-4]), dim=1)
        xb = self.REA1(xr, xb)
        xb = self.boundary_decode_stages[1](xb)

        return xb, xr

# EOF