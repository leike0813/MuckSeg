from lib.lightning_framework import DEFAULT_ENVIRONMENT
from lib.lightning_framework.trainer import DEFAULT_CONFIG as DEFAULT_CONFIG_TRAIN, EXAMPLE_MODEL_CONFIG
from lib.lightning_framework.metrics import DEFAULT_CONFIG_SEGMENTATION as DEFAULT_CONFIG_METRICS
from lib.lightning_framework.callbacks import DEFAULT_CONFIG_PREDICTION_WRITER
from lib.pytorch_framework.utils import CustomCfgNode as CN
from lib.pytorch_framework.utils import update_config
from lib.pytorch_framework.loss_functions import DEFAULT_CONFIG as DEFAULT_CONFIG_LOSS_FUNCTION
from lib.pytorch_framework.optimizers.default_config import DEFAULT_CONFIG as DEFAULT_CONFIG_OPTIMIZER
from lib.pytorch_framework.lr_schedulers import DEFAULT_CONFIG as DEFAULT_CONFIG_LR_SCHEDULER
from lib.pytorch_framework.visualization import DEFAULT_CONFIG as DEFAULT_CONFIG_VISUALIZATION
from utils.post_processing import DEFAULT_CONFIG as DEFAULT_CONFIG_POST_PROCESSOR


_C = CN()
_C.register_deprecated_key('PREDICT.RESULT_PATH')
_C.register_deprecated_key('TRAIN_STAGE1.LOSS_FUNC.SIDE_OUTPUT_SUPERVISION_TYPES')
_C.register_deprecated_key('TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.SIDE_OUTPUT_SUPERVISION_TYPES')
_C.register_deprecated_key('TRAIN_STAGE2.LOSS_FUNC.REGION.SIDE_OUTPUT_SUPERVISION_TYPES')

_C.BASE = ['']

_C.ENVIRONMENT = DEFAULT_ENVIRONMENT.clone()
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = EXAMPLE_MODEL_CONFIG.clone()
_C.MODEL.ABBR.MuckSeg = 'SS'
# Model type
# could be overwritten by command line argument
_C.MODEL.TYPE = 'MuckSeg'
# Model name
_C.MODEL.SPEC_NAME = '512_D32'
_C.MODEL.FILE_PATHS = [
    'models/__init__.py',
    'models/build.py',
    'models/MuckSeg_Encoder.py',
    'models/MuckSeg_Decoder_Stage1.py',
    'models/MuckSeg_Decoder_Stage2.py',
    'models/MuckSeg_Head.py',
    'models/MuckSeg_Neck.py',
    'models/MuckSeg.py',
    'lightning_module/module.py',
]

_C.MODEL.IMAGE_SIZE = 512
_C.MODEL.IN_CHANS = 1
_C.MODEL.OUT_CHANS = 1
_C.MODEL.DIM = 32
_C.MODEL.MLP_RATIO = 4.
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.TASK = 'binary'
_C.MODEL.USE_CONVNEXT_V2 = True

_C.MODEL.set_invisible_keys(['IN_CHANS', 'OUT_CHANS', 'TASK'])

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.kernel_sizes = [7, 7, 7, 7]
_C.MODEL.ENCODER.depths = [3, 9, 3, 3]
_C.MODEL.ENCODER.stem_routes = ['3CONV', '5CONV', '7CONV', '9CONV']
_C.MODEL.ENCODER.multi_scale_input = False

_C.MODEL.DECODER_STAGE1 = CN()
_C.MODEL.DECODER_STAGE1.kernel_sizes = [7, 7]
_C.MODEL.DECODER_STAGE1.depths = [2, 2]

_C.MODEL.DECODER_STAGE2 = CN()
_C.MODEL.DECODER_STAGE2.boundary_kernel_sizes = [7, 7]
_C.MODEL.DECODER_STAGE2.boundary_depths = [2, 2]
_C.MODEL.DECODER_STAGE2.region_kernel_sizes = [7, 7]
_C.MODEL.DECODER_STAGE2.region_depths = [2, 2]
_C.MODEL.DECODER_STAGE2.rea_kernel_size = 7

_C.MODEL.NECK = CN()
_C.MODEL.NECK.kernel_size = 3
_C.MODEL.NECK.depth = 2
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
# Stage 1
_C.TRAIN_STAGE1 = DEFAULT_CONFIG_TRAIN.clone()
_C.TRAIN_STAGE1.BATCH_SIZE = 10
_C.TRAIN_STAGE1.USE_BATCHSIZE_FINDER = False
_C.TRAIN_STAGE1.MONITOR = 'valid/epoch/IoU'
_C.TRAIN_STAGE1.MONITOR_MODE = 'max'
# Callbacks
# Whether to use custom checkpointing callback
_C.TRAIN_STAGE1.USE_CUSTOM_CHECKPOINTING = True
_C.TRAIN_STAGE1.CHECKPOINT_TOPK = 1
_C.TRAIN_STAGE1.CHECKPOINT_SAVELAST = True
# Whether to use earlystopping
# could be overwritten by command line argument
_C.TRAIN_STAGE1.USE_EARLYSTOPPING = False
_C.TRAIN_STAGE1.EARLYSTOPPING_PATIENCE = 10

_C.TRAIN_STAGE1.FCMAE_CKPT_PATH = ''
# Experiment name to be logged
# could be overwritten by command line argument
_C.TRAIN_STAGE1.EXPERIMENT_NAME = 'MuckSeg'
# Experiment tag to be used as part of run name
# could be overwritten by command line argument
_C.TRAIN_STAGE1.TAG = ''

# Loss function
_C.TRAIN_STAGE1.LOSS_FUNC = DEFAULT_CONFIG_LOSS_FUNCTION.clone()
# Loss function to be used for stage 1
_C.TRAIN_STAGE1.LOSS_FUNC.NAME = 'Raw_Region'
_C.TRAIN_STAGE1.LOSS_FUNC.FUNC_NAME = 'Compose'
_C.TRAIN_STAGE1.LOSS_FUNC.WEIGHT = 1.0
_C.TRAIN_STAGE1.LOSS_FUNC.SIDE_OUTPUT_WEIGHTS = [0.7, 0.3]
# _C.TRAIN_STAGE1.LOSS_FUNC.SIDE_OUTPUT_SUPERVISION_TYPES = ['bin', 'bin']
_C.TRAIN_STAGE1.LOSS_FUNC.COMPOSE.LIST = [
    'BCELoss',
    'BinaryFocalLoss',
]
_C.TRAIN_STAGE1.LOSS_FUNC.COMPOSE.WEIGHT = [0.4, 0.6]

_C.TRAIN_STAGE1.LOSS_FUNC.BCELOSS.pos_weight = [0.8]
_C.TRAIN_STAGE1.LOSS_FUNC.BINARYFOCALLOSS.alpha = 0.4

# Optimizer
_C.TRAIN_STAGE1.OPTIMIZER = DEFAULT_CONFIG_OPTIMIZER.clone()
_C.TRAIN_STAGE1.OPTIMIZER.BASE_LR = 1e-3

# LR scheduler
_C.TRAIN_STAGE1.LR_SCHEDULER = DEFAULT_CONFIG_LR_SCHEDULER.clone()
_C.TRAIN_STAGE1.LR_SCHEDULER.MONITOR = _C.TRAIN_STAGE1.MONITOR
_C.TRAIN_STAGE1.LR_SCHEDULER.MONITOR_MODE = _C.TRAIN_STAGE1.MONITOR_MODE

# Metrics
_C.TRAIN_STAGE1.METRICS = CN(visible=False)
_C.TRAIN_STAGE1.METRICS.RAW_REGION = DEFAULT_CONFIG_METRICS.clone()
_C.TRAIN_STAGE1.METRICS.RAW_REGION.CATEGORY = 'Raw_Region'
_C.TRAIN_STAGE1.METRICS.RAW_REGION.APPEND_CATEGORY_NAME = False

# -----------------------------------------------------------------------------
# Stage 2
_C.TRAIN_STAGE2 = DEFAULT_CONFIG_TRAIN.clone()
_C.TRAIN_STAGE2.BATCH_SIZE = 8
_C.TRAIN_STAGE2.USE_BATCHSIZE_FINDER = False
_C.TRAIN_STAGE2.MONITOR = 'valid/epoch/IoU_Boundary'
_C.TRAIN_STAGE2.MONITOR_MODE = 'max'
# Callbacks
# Whether to use custom checkpointing callback
_C.TRAIN_STAGE2.USE_CUSTOM_CHECKPOINTING = True
_C.TRAIN_STAGE2.CHECKPOINT_TOPK = 3
_C.TRAIN_STAGE2.CHECKPOINT_SAVELAST = True
# Whether to use earlystopping
# could be overwritten by command line argument
_C.TRAIN_STAGE2.USE_EARLYSTOPPING = False
_C.TRAIN_STAGE2.EARLYSTOPPING_PATIENCE = 10

# Experiment name to be logged
# could be overwritten by command line argument
_C.TRAIN_STAGE2.EXPERIMENT_NAME = 'MuckSeg'
# Experiment tag to be used as part of run name
# could be overwritten by command line argument
_C.TRAIN_STAGE2.TAG = ''

# Loss function
_C.TRAIN_STAGE2.LOSS_FUNC = CN()
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY = DEFAULT_CONFIG_LOSS_FUNCTION.clone()
_C.TRAIN_STAGE2.LOSS_FUNC.REGION = DEFAULT_CONFIG_LOSS_FUNCTION.clone()
# Loss function to be used for boundary branch
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.NAME = 'Boundary'
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.WEIGHT = 0.6
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.SIDE_OUTPUT_WEIGHTS = [0.7, 0.3]
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.FUNC_NAME = 'Compose'
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.COMPOSE.LIST = [
    'BCELoss',
    'BinaryFocalLoss',
]
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.COMPOSE.WEIGHT = [0.2, 0.8]

_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.BCELOSS.pos_weight = [1.6]
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.BINARYFOCALLOSS.alpha = 0.6
_C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.BINARYFOCALLOSS.gamma = 1.0
# _C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.FUNC_NAME = 'SmoothL1NormalizedBCELoss'
# _C.TRAIN_STAGE2.LOSS_FUNC.BOUNDARY.SLN_BCELOSS.pos_weight = [1.6]


# Loss function to be used for region branch
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.NAME = 'Region'
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.WEIGHT = 0.4
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.SIDE_OUTPUT_WEIGHTS = [0.7, 0.3]
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.FUNC_NAME = 'Compose'
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.COMPOSE.LIST = [
    'BCELoss',
    'BinaryFocalLoss',
]
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.COMPOSE.WEIGHT = [0.2, 0.8]

_C.TRAIN_STAGE2.LOSS_FUNC.REGION.BCELOSS.pos_weight = [0.8]
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.BINARYFOCALLOSS.alpha = 0.4
_C.TRAIN_STAGE2.LOSS_FUNC.REGION.BINARYFOCALLOSS.gamma = 1.0
# _C.TRAIN_STAGE2.LOSS_FUNC.REGION.FUNC_NAME = 'SmoothL1NormalizedBCELoss'
# _C.TRAIN_STAGE2.LOSS_FUNC.REGION.SLN_BCELOSS.pos_weight = [0.8]

# Optimizer

_C.TRAIN_STAGE2.OPTIMIZER = DEFAULT_CONFIG_OPTIMIZER.clone()
_C.TRAIN_STAGE2.OPTIMIZER.BASE_LR = 2e-4

# LR scheduler
_C.TRAIN_STAGE2.LR_SCHEDULER = DEFAULT_CONFIG_LR_SCHEDULER.clone()
_C.TRAIN_STAGE2.LR_SCHEDULER.MONITOR = _C.TRAIN_STAGE2.MONITOR
_C.TRAIN_STAGE2.LR_SCHEDULER.MONITOR_MODE = _C.TRAIN_STAGE2.MONITOR_MODE

# Metrics
_C.TRAIN_STAGE2.METRICS = CN()
_C.TRAIN_STAGE2.METRICS.BOUNDARY = DEFAULT_CONFIG_METRICS.clone()
_C.TRAIN_STAGE2.METRICS.BOUNDARY.CATEGORY = 'Boundary'
_C.TRAIN_STAGE2.METRICS.BOUNDARY.APPEND_CATEGORY_NAME = True

_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Precision = CN(visible=False)
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Precision.task = 'binary'
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Precision.num_classes = 1
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Precision.LEVEL = 2

_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Recall = CN(visible=False)
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Recall.task = 'binary'
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Recall.num_classes = 1
_C.TRAIN_STAGE2.METRICS.BOUNDARY.VALIDATION.Recall.LEVEL = 2

_C.TRAIN_STAGE2.METRICS.REGION = DEFAULT_CONFIG_METRICS.clone()
_C.TRAIN_STAGE2.METRICS.REGION.CATEGORY = 'Region'
_C.TRAIN_STAGE2.METRICS.REGION.APPEND_CATEGORY_NAME = True

_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Precision = CN(visible=False)
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Precision.task = 'binary'
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Precision.num_classes = 1
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Precision.LEVEL = 2

_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Recall = CN(visible=False)
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Recall.task = 'binary'
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Recall.num_classes = 1
_C.TRAIN_STAGE2.METRICS.REGION.VALIDATION.Recall.LEVEL = 2
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Dataset name
_C.DATA.DATA_PATH = ''
_C.DATA.DATAMODULE = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.DATAMODULE.batch_size = _C.TRAIN_STAGE1.BATCH_SIZE
_C.DATA.DATAMODULE.val_volume = 100
_C.DATA.DATAMODULE.test_volume = 1000
# Path to dataset, could be overwritten by command line argument
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.DATAMODULE.pin_memory = True
# Number of data loading threads
_C.DATA.DATAMODULE.num_workers = 0
# Whether to shuffle data in train_loader
_C.DATA.DATAMODULE.shuffle = True
# If explicitly given, the partition of dataset for each experiment instance will be fixed
_C.DATA.DATAMODULE.split_seed = 33

_C.DATA.DATAMODULE.set_invisible_keys(['pin_memory', 'shuffle', 'split_seed'])
# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.PREDICT = CN()
# Path to output results, could be overwritten by command line argument
_C.PREDICT.DATA_PATH = ''
_C.PREDICT.CKPT_PATH = ''
# Original image size in W x H format (opencv fashion)
_C.PREDICT.IMAGE_SIZE = (2048, 4096)
# Original image ROI in (left, top, width, height) format (opencv fashion)
_C.PREDICT.IMAGE_ROI = (256, 0, 1536, 4096)
_C.PREDICT.THREEFOLD_MARGIN_RATE = 0.1
_C.PREDICT.IMAGE_MEAN = [0.618]
_C.PREDICT.IMAGE_STD = [0.229]

_C.PREDICT.set_invisible_keys(['CKPT_PATH', 'THREEFOLD_MARGIN_RATE'])

_C.PREDICT.DATAMODULE = CN(visible=False)
_C.PREDICT.DATAMODULE.batch_size = 1
_C.PREDICT.DATAMODULE.pin_memory = True
# Number of data loading threads
_C.PREDICT.DATAMODULE.num_workers = 0

_C.PREDICT.DATAMODULE.set_invisible_keys(['pin_memory'])

_C.PREDICT.WRITER = DEFAULT_CONFIG_PREDICTION_WRITER.clone()
_C.PREDICT.POST_PROCESSING = DEFAULT_CONFIG_POST_PROCESSOR.clone()
# -----------------------------------------------------------------------------
# Visualization settings
# -----------------------------------------------------------------------------
_C.VISUALIZATION = DEFAULT_CONFIG_VISUALIZATION.clone()
_C.VISUALIZATION.IMAGE_CHANNELS = _C.MODEL.IN_CHANS
_C.VISUALIZATION.NUM_CLASSES = _C.MODEL.OUT_CHANS
_C.VISUALIZATION.IMAGE_MEAN = _C.PREDICT.IMAGE_MEAN
_C.VISUALIZATION.IMAGE_STD = _C.PREDICT.IMAGE_STD
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
_C.CONFIG_OUTPUT_PATH = 'configs'
_C.FULL_DUMP = False

_C.set_invisible_keys(['FULL_DUMP'])


def get_config(args, arg_mapper):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, arg_mapper)

    return config

# EOF