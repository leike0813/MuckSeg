import lightning as L


class MuckSeg(L.LightningModule):
    """MuckSeg module.
       Works as a container to assemble the network modules.

    Args:
        encoder (lightning.LightningModule): MuckSeg encoder.
        decoder_stage1 (lightning.LightningModule): MuckSeg decoder stage 1.
        decoder_stage2 (lightning.LightningModule): MuckSeg decoder stage 2.
        neck (lightning.LightningModule or torch.nn.Module): MuckSeg neck.
        head_stage1 (lightning.LightningModule): MuckSeg head stage 1, used for 1st-stage training.
        head_stage2 (lightning.LightningModule): MuckSeg head stage 2.
    """
    is_auto_scalable = True
    size_modulus = 16

    def __init__(self, encoder, decoder_stage1, decoder_stage2, neck, head_stage1, head_stage2):
        super().__init__()
        self.encoder = encoder
        self.decoder_stage1 = decoder_stage1
        self.decoder_stage2 = decoder_stage2
        self.neck = neck
        self.head_stage1 = head_stage1
        self.head_stage2 = head_stage2

        self.__STAGE = 1
        self.decoder_stage2.freeze()
        self.head_stage2.freeze()

    def advance_stage(self):
        assert self.__STAGE == 1, f'The model must be at stage 1 to proceed advancing, but current stage is {self.__STAGE}'
        self.__STAGE = 2
        self.encoder.freeze()
        # self.decoder_stage1.freeze()
        self.decoder_stage1.detach_sideoutput()
        self.head_stage1.freeze()
        self.decoder_stage2.unfreeze()
        self.head_stage2.unfreeze()
        self.forward = self.forward_stage2
        self.forward_featuremaps = self.forward_featuremaps_stage2

    def advance_finetune(self):
        assert self.__STAGE == 2, f'The model must be at stage 2 before fine-tuning, but current stage is {self.__STAGE}'
        self.encoder.unfreeze()
        self.decoder_stage2.detach_sideoutput()
        self.head_stage2.detach_sideoutput()
        self.forward = self.forward_finetune
        self.forward_featuremaps = self.forward_featuremaps_finetune

    def forward(self, x):
        x, xi = self.encoder(x)
        x = self.neck(x)
        x, x0 = self.decoder_stage1(x, xi)

        return self.head_stage1(x, x0)

    def forward_stage2(self, x):
        x, xi = self.encoder(x)
        x = self.neck(x)
        x = self.decoder_stage1(x, xi)
        xb, xr, xb0, xr0 = self.decoder_stage2(x, xi)

        return self.head_stage2(xb, xr, xb0, xr0)

    def forward_finetune(self, x):
        x, xi = self.encoder(x)
        x = self.neck(x)
        x = self.decoder_stage1(x, xi)
        xb, xr = self.decoder_stage2(x, xi)

        return self.head_stage2(xb, xr)


    def forward_featuremaps(self, x):
        fmaps = {}
        x, xi = self.encoder(x)
        for i in range(len(xi)):
            fmaps['Encoder-Stage{si}'.format(si=i)] = xi[i]
        x = self.neck(x)
        fmaps['Neck'] = x
        x, x0 = self.decoder_stage1(x, xi)
        fmaps['Decoder-Stage{si}'.format(si=0)] = x0
        fmaps['Decoder-Stage{si}'.format(si=1)] = x

        return fmaps

    def forward_featuremaps_stage2(self, x):
        fmaps = {}
        x, xi = self.encoder(x)
        for i in range(len(xi)):
            fmaps['Encoder-Stage{si}'.format(si=i)] = xi[i]
        x = self.neck(x)
        fmaps['Neck'] = x
        x = self.decoder_stage1(x, xi)
        fmaps['Decoder-Stage{si}'.format(si=1)] = x
        xb, xr, xb0, xr0 = self.decoder_stage2(x, xi)
        fmaps['Decoder-Boundary-Stage{si}'.format(si=0)] = xb0
        fmaps['Decoder-Boundary-Stage{si}'.format(si=1)] = xb
        fmaps['Decoder-Region-Stage{si}'.format(si=0)] = xr0
        fmaps['Decoder-Region-Stage{si}'.format(si=1)] = xr

        return fmaps

    def forward_featuremaps_finetune(self, x):
        fmaps = {}
        x, xi = self.encoder(x)
        for i in range(len(xi)):
            fmaps['Encoder-Stage{si}'.format(si=i)] = xi[i]
        x = self.neck(x)
        fmaps['Neck'] = x
        x = self.decoder_stage1(x, xi)
        fmaps['Decoder-Stage{si}'.format(si=1)] = x
        xb, xr = self.decoder_stage2(x, xi)
        fmaps['Decoder-Boundary-Stage{si}'.format(si=1)] = xb
        fmaps['Decoder-Region-Stage{si}'.format(si=1)] = xr

        return fmaps

# EOF