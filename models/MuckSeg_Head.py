import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class MuckSeg_Head_Stage1(L.LightningModule):
    def __init__(self, out_chans=1, dim=128, dim_side=256):
        super().__init__()
        self.head = nn.Conv2d(dim, out_chans, kernel_size=1)
        self.head_side = nn.Conv2d(dim_side, out_chans, kernel_size=1)

    def forward(self, x, x0):
        return self.head(x), F.interpolate(self.head_side(x0), scale_factor=2, mode="bilinear")


class MuckSeg_Head_Stage2(L.LightningModule):
    def __init__(self, out_chans=1, dim=32, dim_side=64):
        super().__init__()
        self.boundary_head = nn.Conv2d(dim, out_chans, kernel_size=1)
        self.boundary_head_side = nn.Conv2d(dim_side, out_chans, kernel_size=1)
        self.region_head = nn.Conv2d(dim, out_chans, kernel_size=1)
        self.region_head_side = nn.Conv2d(dim_side, out_chans, kernel_size=1)
        self.side_output_detached = False

    def detach_sideoutput(self):
        self.forward = self.forward_detached
        self.side_output_detached = True

    def forward(self, xb, xr, xb0, xr0):
        return self.boundary_head(xb), self.region_head(xr),\
            F.interpolate(self.boundary_head_side(xb0), scale_factor=2, mode="bilinear"),\
            F.interpolate(self.region_head_side(xr0), scale_factor=2, mode="bilinear")

    def forward_detached(self, xb, xr):
        return self.boundary_head(xb), self.region_head(xr)

# EOF