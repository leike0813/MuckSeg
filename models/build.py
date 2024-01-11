from pathlib import Path
from torch import Tensor
from .MuckSeg_Encoder import MuckSeg_Encoder
from .MuckSeg_Decoder_Stage1 import MuckSeg_Decoder_Stage1
from .MuckSeg_Decoder_Stage2 import MuckSeg_Decoder_Stage2
from .MuckSeg_Neck import MuckSeg_Neck
from .MuckSeg_Head import MuckSeg_Head_Stage1, MuckSeg_Head_Stage2
from .MuckSeg import MuckSeg


__all__ = ['build_model']


def build_model(node):
    node.defrost()
    cwd = Path.cwd()
    node.FILE_PATHS = [(cwd / path).as_posix() for path in node.FILE_PATHS]
    node.freeze()

    encoder = MuckSeg_Encoder(
        in_chans=node.IN_CHANS,
        dim=node.DIM,
        mlp_ratio=node.MLP_RATIO,
        drop_path_rate=node.DROP_PATH_RATE,
        use_convnext_v2=node.USE_CONVNEXT_V2,
        **node.ENCODER
    )
    hparams = {
        'image-size': node.IMAGE_SIZE,
        'input-channel': node.IN_CHANS,
        'output-channel': node.OUT_CHANS,
        'embedding-dim': node.DIM,
        'mlp-ratio': node.MLP_RATIO,
        'encoder-kernel-sizes': node.ENCODER.kernel_sizes,
        'encoder-depths': node.ENCODER.depths,
        'multi-scale-input': node.ENCODER.multi_scale_input,
    }
    decoder_stage1 = MuckSeg_Decoder_Stage1(
        dim=node.DIM * 2 ** MuckSeg_Encoder._NUM_TIERS,
        mlp_ratio=node.MLP_RATIO,
        drop_path_rate=node.DROP_PATH_RATE / 2,
        use_convnext_v2=node.USE_CONVNEXT_V2,
        **node.DECODER_STAGE1
    )
    hparams.update({
        'decoder-S1-kernel-sizes': node.DECODER_STAGE1.kernel_sizes,
        'decoder-S1-depths': node.DECODER_STAGE1.depths,
    })
    decoder_stage2 = MuckSeg_Decoder_Stage2(
        dim=node.DIM * 2 ** (MuckSeg_Encoder._NUM_TIERS - 2),
        mlp_ratio=node.MLP_RATIO,
        drop_path_rate=[node.DROP_PATH_RATE / 2, node.DROP_PATH_RATE],
        use_convnext_v2=node.USE_CONVNEXT_V2,
        **node.DECODER_STAGE2
    )
    hparams.update({
        'decoder-S2-boundary-kernel-sizes': node.DECODER_STAGE2.boundary_kernel_sizes,
        'decoder-S2-boundary-depths': node.DECODER_STAGE2.boundary_depths,
        'decoder-S2-region-kernel-sizes': node.DECODER_STAGE2.region_kernel_sizes,
        'decoder-S2-region-depths': node.DECODER_STAGE2.region_depths,
        'reverse-edge-attention-kernel-size': node.DECODER_STAGE2.rea_kernel_size
    })
    neck = MuckSeg_Neck(
        dim=node.DIM * 2 ** MuckSeg_Encoder._NUM_TIERS,
        mlp_ratio=node.MLP_RATIO,
        drop_path=node.DROP_PATH_RATE,
        use_convnext_v2=node.USE_CONVNEXT_V2,
        **node.NECK
    )
    hparams.update({
        'neck-kernel-size': node.NECK.kernel_size,
        'neck-depth': node.NECK.depth,
    })
    head_stage1 = MuckSeg_Head_Stage1(
        out_chans=node.OUT_CHANS,
        dim=node.DIM * 2 ** (MuckSeg_Encoder._NUM_TIERS - 2),
        dim_side=node.DIM * 2 ** (MuckSeg_Encoder._NUM_TIERS - 1),
    )
    head_stage2 = MuckSeg_Head_Stage2(
        out_chans=node.OUT_CHANS,
        dim=node.DIM,
        dim_side=node.DIM * 2,
    )

    return MuckSeg(
        encoder=encoder,
        decoder_stage1=decoder_stage1,
        decoder_stage2=decoder_stage2,
        neck=neck,
        head_stage1=head_stage1,
        head_stage2=head_stage2,
    ), hparams, Tensor(
        1,
        node.IN_CHANS,
        node.IMAGE_SIZE,
        node.IMAGE_SIZE,
    )

# EOF