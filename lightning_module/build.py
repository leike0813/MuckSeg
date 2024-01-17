from lib.pytorch_framework.optimizers import build_optimizer
from lib.pytorch_framework.lr_schedulers import build_scheduler
from lib.pytorch_framework.loss_functions import build_lossfunc
from models import build_model
from .module import MuckSeg_Lightning_Module
from .proxy import build_lossfunc_stage2, build_visualizer
from utils.post_processing import build_post_processor


def build_lightning_module(config):
    model, model_hparams, example_input = build_model(config.MODEL)

    _, loss_fn_stage1, loss_hparams_stage1 = build_lossfunc(config.TRAIN_STAGE1.LOSS_FUNC)
    optimizer_stage1, opt_hparams_stage1 = build_optimizer(model, config.TRAIN_STAGE1.OPTIMIZER)
    scheduler_config_stage1, sche_params_stage1 = build_scheduler(optimizer_stage1, config.TRAIN_STAGE1.LR_SCHEDULER)
    hparams_stage1 = model_hparams.copy()
    hparams_stage1.update(loss_hparams_stage1)
    hparams_stage1.update(opt_hparams_stage1)
    hparams_stage1.update(sche_params_stage1)

    loss_fn_finetune, loss_fn_stage2, loss_hparams_stage2 = build_lossfunc_stage2(config.TRAIN_STAGE2.LOSS_FUNC)
    optimizer_stage2, opt_hparams_stage2 = build_optimizer(model, config.TRAIN_STAGE2.OPTIMIZER)
    scheduler_config_stage2, sche_params_stage2 = build_scheduler(optimizer_stage2, config.TRAIN_STAGE2.LR_SCHEDULER)
    test_visualizer, featuremap_visualizer = build_visualizer(config)
    hparams_stage2 = model_hparams.copy()
    hparams_stage2.update(loss_hparams_stage2)
    hparams_stage2.update(opt_hparams_stage2)
    hparams_stage2.update(sche_params_stage2)

    post_processor = build_post_processor(config.PREDICT.POST_PROCESSING)

    return MuckSeg_Lightning_Module(
        model=model,
        loss_fn_stage1=loss_fn_stage1,
        loss_fn_stage2=loss_fn_stage2,
        loss_fn_finetune=loss_fn_finetune,
        optimizer_stage1=optimizer_stage1,
        optimizer_stage2=optimizer_stage2,
        scheduler_stage1=scheduler_config_stage1,
        scheduler_stage2=scheduler_config_stage2,
        test_visualizer=test_visualizer,
        post_processor=post_processor,
        featuremap_visualizer=featuremap_visualizer,
        config=config,
        example_input=example_input,
    ), hparams_stage1, hparams_stage2