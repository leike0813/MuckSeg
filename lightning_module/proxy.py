import os
from pathlib import Path
import json
from lib.pytorch_framework.loss_functions import build_lossfunc
from lib.pytorch_framework.visualization import build_visualizer as build_visualizer_atom
from lib.lightning_framework.callbacks import build_prediction_writer as build_prediction_writer_atom
from lib.lightning_framework.trainer import build_trainer as build_trainer_atom
from lib.lightning_framework.trainer import DEFAULT_CONFIG as DEFAULT_CONFIG_TRAINER, EXAMPLE_MODEL_CONFIG
from lib.lightning_framework import DEFAULT_ENVIRONMENT


__all__ = [
    'build_trainer',
    'build_prediction_writer',
    'build_visualizer',
    'build_lossfunc_stage2',
]


def build_trainer(train_node=DEFAULT_CONFIG_TRAINER, model_node=EXAMPLE_MODEL_CONFIG, env_node=DEFAULT_ENVIRONMENT, extra_callbacks=[]):
    trainer, hparams = build_trainer_atom(train_node, model_node, env_node, extra_callbacks)
    if os.path.isfile(os.path.join(env_node.DATA_BASE_PATH, train_node.FCMAE_CKPT_PATH)):
        hparams.update({
            'fcmae_checkpoint': os.path.join(env_node.DATA_BASE_PATH, train_node.FCMAE_CKPT_PATH)
        })
    return trainer, hparams


def build_prediction_writer(node, mode='train'):
    node.defrost()
    if mode == 'train':
        node.concatenate = 2
        node.log_prediction = True
        node.log_folder = 'inference_example'
    elif mode == 'inference':
        pass
    node.freeze()
    return build_prediction_writer_atom(node)


def build_visualizer(config):
    try:
        with open((Path(config.ENVIRONMENT.DATA_BASE_PATH) / config.DATA.DATA_PATH) / 'statistics.json', 'r') as f:
            train_statistics = json.load(f)
        config.defrost()
        config.VISUALIZATION.IMAGE_MEAN = train_statistics['mean']
        config.VISUALIZATION.IMAGE_STD = train_statistics['std']
        config.freeze()
    except Exception:
        pass

    return build_visualizer_atom(config.VISUALIZATION)


def build_lossfunc_stage2(stage2_node):
    loss_fn_boundary, loss_fn_side_boundary, loss_hparams_boundary = build_lossfunc(stage2_node.BOUNDARY)
    loss_fn_region, loss_fn_side_region, loss_hparams_region = build_lossfunc(stage2_node.REGION)
    loss_hparams = loss_hparams_boundary
    loss_hparams.update(loss_hparams_region)

    def loss_func_with_sideoutput(y_pred, y_bound, y_region):
        y_pred_bound = []
        y_pred_region = []
        for i in range(len(y_pred) // 2):
            y_pred_bound.append(y_pred[i * 2])
            y_pred_region.append(y_pred[i * 2 + 1])
        return loss_fn_side_boundary(y_pred_bound, y_bound) + loss_fn_side_region(y_pred_region, y_region)

    def loss_func(y_pred, y_bound, y_region):
        y_pred_bound = [y_pred[0]]
        y_pred_region = [y_pred[1]]
        return loss_fn_boundary(y_pred_bound, y_bound) + loss_fn_region(y_pred_region, y_region)

    return loss_func, loss_func_with_sideoutput, loss_hparams

# EOF