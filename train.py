import warnings
import argparse
from pathlib import Path
from config import get_config
import torch
from lightning_module import build_lightning_module
from data import build_datamodule, build_inference_dataloader
from lib.lightning_framework.trainer import build_trainer
from lightning_module.proxy import build_prediction_writer, build_trainer as build_trainer_stage1
from lib.lightning_framework.metrics import build_metric_callback


def parse_option():
    parser = argparse.ArgumentParser('MuckSeg training and evaluation script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    group.add_argument('--resume-from-run-path', type=str,
                       help="path to folder which saves model configuration and checkpoint to resume training")

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--stage', choices=[1, 2], type=int, default=1, help="training stage to starts with")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--max-epochs', type=int, help="maximum training epochs")
    parser.add_argument('--min-epochs', type=int, help="minimum training epochs")
    parser.add_argument('--base-lr', type=float, help="base learning rate for training")
    parser.add_argument('--overfit-batches', type=int, help="overfit batches for testing or finetuning")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--checkpoint-path', type=str, help='path to model checkpoint')
    parser.add_argument('--fcmae-run-path', type=str, help='path to FCMAE run folder which saves pretraining weights and configuration')
    parser.add_argument('--fcmae-checkpoint-path', type=str, help='path to FCMAE chechpoint (without validation of model configuration)')
    parser.add_argument('--extra-cfg', type=str, metavar="FILE", help='path to extra config file')
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--spec-name', type=str, help='model spec name')
    parser.add_argument('--use-earlystopping', action='store_true',
                        help="whether to use early stopping to prevent overfitting")
    parser.add_argument('--use-custom-checkpointing', action='store_true',
                        help="whether to use custom checkpointing callback")
    parser.add_argument('--device', type=str, help='accelerate device to be used')
    parser.add_argument('--config-output-path', type=str, metavar='PATH',
                        help='root of output folder, the full path is <config-output-path>/<model_type>_<model_spec>_<tag>.yaml')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--logger', type=str,
                        help='overwrite logger if provided, can be MLFlowLogger only for now.')
    parser.add_argument('--full-dump', action='store_true',
                        help="whether to dump full config file")
    parser.add_argument('--ignore-warnings', action='store_true',
                        help="whether to ignore warnings during execution")

    parser.add_argument('--s2-batch-size', type=int, help="batch size for single GPU (stage 2)")
    parser.add_argument('--s2-max-epochs', type=int, help="maximum training epochs (stage 2)")
    parser.add_argument('--s2-min-epochs', type=int, help="minimum training epochs (stage 2)")
    parser.add_argument('--s2-base-lr', type=float, help="base learning rate for training (stage 2)")
    parser.add_argument('--s2-overfit-batches', type=int, help="overfit batches for testing or finetuning (stage 2")
    parser.add_argument('--s2-use-earlystopping', action='store_true',
                        help="whether to use early stopping to prevent overfitting (stage 2)")
    parser.add_argument('--s2-use-custom-checkpointing', action='store_true',
                        help="whether to use custom checkpointing callback (stage 2)")

    args, unparsed = parser.parse_known_args()

    if args.stage == 1:
        arg_mapper = {
            'batch_size': {'TRAIN_STAGE1.BATCH_SIZE': None, 'TRAIN_STAGE1.USE_BATCHSIZE_FINDER': False},
            'max_epochs': {'TRAIN_STAGE1.TRAINER.max_epochs': None},
            'min_epochs': {'TRAIN_STAGE1.TRAINER.min_epochs': None},
            'overfit_batches': {'TRAIN_STAGE1.TRAINER.overfit_batches': None},
            'base_lr': {'TRAIN_STAGE1.OPTIMIZER.BASE_LR': None},
            'data_path': {'DATA.DATA_PATH': None},
            'experiment': {'TRAIN_STAGE1.EXPERIMENT_NAME': None, 'TRAIN_STAGE2.EXPERIMENT_NAME': None},
            'spec_name': {'MODEL.SPEC_NAME': None},
            'use_earlystopping': {'TRAIN_STAGE1.USE_EARLYSTOPPING': None},
            'use_custom_checkpointing': {'TRAIN_STAGE1.USE_CUSTOM_CHECKPOINTING': None},
            'device': {'TRAIN_STAGE1.TRAINER.accelerator': None, 'TRAIN_STAGE2.TRAINER.accelerator': None},
            'config_output_path': {'CONFIG_OUTPUT_PATH': None},
            'tag': {'TRAIN_STAGE1.TAG': '{}{}'.format(args.tag, 'Stage1' if args.tag is None else '_Stage1'),
                    'TRAIN_STAGE2.TAG': '{}{}'.format(args.tag, 'Stage2' if args.tag is None else '_Stage2')},
            'logger': {'TRAIN_STAGE1.LOGGER.NAME': None, 'TRAIN_STAGE2.LOGGER.NAME': None},
            'full_dump': {'FULL_DUMP': None},
            's2_batch_size': {'TRAIN_STAGE2.BATCH_SIZE': None, 'TRAIN_STAGE2.USE_BATCHSIZE_FINDER': False},
            's2_max_epochs': {'TRAIN_STAGE2.TRAINER.max_epochs': None},
            's2_min_epochs': {'TRAIN_STAGE2.TRAINER.min_epochs': None},
            's2_overfit_batches': {'TRAIN_STAGE2.TRAINER.overfit_batches': None},
            's2_base_lr': {'TRAIN_STAGE2.OPTIMIZER.BASE_LR': None},
            's2_use_earlystopping': {'TRAIN_STAGE2.USE_EARLYSTOPPING': None},
            's2_use_custom_checkpointing': {'TRAIN_STAGE2.USE_CUSTOM_CHECKPOINTING': None},
            'checkpoint_path': {'TRAIN_STAGE1.CKPT_PATH': None},
            'fcmae_checkpoint_path': {'TRAIN_STAGE1.FCMAE_CKPT_PATH': None},
        }
    elif args.stage == 2:
        arg_mapper = {
            'batch_size': {'TRAIN_STAGE2.BATCH_SIZE': None, 'TRAIN_STAGE2.USE_BATCHSIZE_FINDER': False},
            'max_epochs': {'TRAIN_STAGE2.TRAINER.max_epochs': None},
            'min_epochs': {'TRAIN_STAGE2.TRAINER.min_epochs': None},
            'overfit_batches': {'TRAIN_STAGE2.TRAINER.overfit_batches': None},
            'base_lr': {'TRAIN_STAGE2.OPTIMIZER.BASE_LR': None},
            'data_path': {'DATA.DATA_PATH': None},
            'experiment': {'TRAIN_STAGE2.EXPERIMENT_NAME': None},
            'spec_name': {'MODEL.SPEC_NAME': None},
            'use_earlystopping': {'TRAIN_STAGE2.USE_EARLYSTOPPING': None},
            'use_custom_checkpointing': {'TRAIN_STAGE2.USE_CUSTOM_CHECKPOINTING': None},
            'device': {'TRAIN_STAGE2.TRAINER.accelerator': None},
            'config_output_path': {'CONFIG_OUTPUT_PATH': None},
            'tag': {'TRAIN_STAGE2.TAG': None},
            'logger': {'TRAIN_STAGE2.LOGGER.NAME': None},
            'full_dump': {'FULL_DUMP': None},
            'checkpoint_path': {'TRAIN_STAGE2.CKPT_PATH': None},
        }

    if args.resume_from_run_path is not None:
        import yaml
        with open('configs/environment._local.yaml', 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            env_settings = [
                'ENVIRONMENT.DATA_BASE_PATH',
                yaml_cfg['ENVIRONMENT']['DATA_BASE_PATH'],
                'ENVIRONMENT.RESULT_BASE_PATH',
                yaml_cfg['ENVIRONMENT']['RESULT_BASE_PATH'],
                'ENVIRONMENT.PROJECT_PATH',
                yaml_cfg['ENVIRONMENT']['PROJECT_PATH'],
                'ENVIRONMENT.MLFLOW_BASE_PATH',
                yaml_cfg['ENVIRONMENT']['MLFLOW_BASE_PATH'],
            ]
            if args.opts is None:
                args.opts = env_settings
            else:
                args.opts.extend(env_settings)

        try:
            run_path = Path(args.resume_from_run_path)
            args.cfg = next(iter(run_path.rglob('*Recent.yaml'))).as_posix()
            checkpoint_fld = next(iter(run_path.rglob('checkpoints')))
            _checkpoint_found = False
            for checkpoint in checkpoint_fld.glob('*'):
                with open(checkpoint / 'aliases.txt', 'r') as f:
                    aliases = f.readline()
                if 'best' in aliases:
                    checkpoint_path = next(iter(checkpoint.glob('*.ckpt'))).as_posix()
                    args.checkpoint_path = checkpoint_path
                    _checkpoint_found = True
            if not _checkpoint_found:
                warnings.warn('Cannot find model checkpoint, the training will start from scratch')
        except Exception as e:
            warnings.warn('Error occurred while loading settings from run folder:\n{msg}'.format(msg=e.__repr__()))
            if args.cfg is None:
                raise FileNotFoundError('Cannot load configuration file')
    else:
        if args.stage == 2:
            warnings.warn('It is necessary to start training stage 2 from a checkpoint, otherwise meaningful results are unlikely to be obtained.', UserWarning)

    config = get_config(args, arg_mapper)

    if args.extra_cfg is not None:
        try:
            config.merge_from_file(args.extra_cfg)
        except FileNotFoundError:
            print('{cfgfile} does not exist'.format(cfgfile=args.extra_cfg))
        except Exception:
            print('Error occurred while loading extra configuration file {cfgfile}'.format(cfgfile=args.extra_cfg))

    if args.fcmae_run_path is not None:
        if args.stage == 2:
            warnings.warn('Can only load FCMAE checkpoint in training stage 1, the fcmae run path assigned will be ignored')
        else:
            try:
                from utils import config_validator
                fcmae_run_path = Path(args.fcmae_run_path)
                fcmae_cfg_path = next(iter(fcmae_run_path.rglob('*Recent.yaml'))).as_posix()
                if not config_validator(args.cfg, fcmae_cfg_path):
                    warnings.warn('FCMAE encoder configuration does not match current configuration, the training will start from scratch')
                    raise ValueError

                checkpoint_fld = next(iter(fcmae_run_path.rglob('checkpoints')))
                _fcmae_checkpoint_found = False
                for checkpoint in checkpoint_fld.glob('*'):
                    if checkpoint.name == 'last':
                        checkpoint_path = next(iter(checkpoint.glob('*.ckpt'))).as_posix()
                        args.fcmae_checkpoint_path = checkpoint_path
                        config.defrost()
                        config.TRAIN_STAGE1.FCMAE_CKPT_PATH = checkpoint_path
                        config.freeze()
                        _fcmae_checkpoint_found = True
                if not _fcmae_checkpoint_found:
                    warnings.warn('Cannot find FCMAE model checkpoint, the training will start from scratch')
            except Exception as e:
                warnings.warn('Error occurred while loading settings from fcmae run folder:\n{msg}'.format(msg=e.__repr__()))

    return args, config


def main(config, stage, ignore_warnings=False):
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    if config.get('TRAIN_STAGE{}'.format(stage)).TRAINER.accelerator == 'cpu':
        config.defrost()
        config.DATA.DATAMODULE.pin_memory = False
        config.PREDICT.DATAMODULE.pin_memory = False
        config.freeze()

    datamodule = build_datamodule(config)
    dataloader = build_inference_dataloader(config)

    tuner = None

    lightningmodule, hparams_stage1, hparams_stage2 = build_lightning_module(config)

    if stage == 1:
        # Stage 1
        print('Initializing training stage 1...')
        metric_callback_stage1 = build_metric_callback(config.TRAIN_STAGE1.METRICS.RAW_REGION)
        prediction_writer_stage1 = build_prediction_writer(config.PREDICT.WRITER, mode='train')
        callbacks_s1 = [metric_callback_stage1, prediction_writer_stage1]

        trainer, train_hparams = build_trainer_stage1(
            train_node=config.TRAIN_STAGE1,
            model_node=config.MODEL,
            env_node=config.ENVIRONMENT,
            extra_callbacks=callbacks_s1
        )

        checkpoint_path = train_hparams.get('checkpoint', None)
        fcmae_checkpoint_path = train_hparams.get('fcmae_checkpoint', None)
        if checkpoint_path is not None:
            print('Loading model parameters from checkpoint: {ckpt}'.format(ckpt=checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            lightningmodule.load_state_dict(checkpoint['state_dict'])
        else:
            if fcmae_checkpoint_path is not None:
                print('Loading FCMAE encoder parameters from checkpoint: {ckpt}'.format(ckpt=fcmae_checkpoint_path))
                checkpoint = torch.load(fcmae_checkpoint_path)
                lightningmodule.load_fcmae(checkpoint['state_dict'])
        hparams_stage1.update(train_hparams)

        if config.TRAIN_STAGE1.USE_BATCHSIZE_FINDER:
            import os
            from lightning.pytorch.tuner import Tuner
            from lightning import Trainer
            temp_trainer = Trainer(
                accelerator=config.TRAIN_STAGE1.TRAINER.accelerator, precision=config.TRAIN_STAGE1.TRAINER.precision,
                default_root_dir=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, 'tuner'),
            )
            tuner = Tuner(temp_trainer)
            batch_size = tuner.scale_batch_size(lightningmodule, datamodule=datamodule, mode='binsearch')
        else:
            batch_size = config.TRAIN_STAGE1.BATCH_SIZE
            datamodule.batch_size = batch_size
        hparams_stage1['batch_size'] = batch_size
        trainer.logger.log_hyperparams(hparams_stage1)

        if config.TRAIN_STAGE1.TRAINER.max_epochs > 0:
            print('Start fitting procedure in training stage 1...')
            trainer.fit(model=lightningmodule, datamodule=datamodule)
            print('Start test procedure in training stage 1...')
            trainer.test(model=lightningmodule, datamodule=datamodule, ckpt_path='best')
        else:
            print('Start test procedure in training stage 1...')
            trainer.test(model=lightningmodule, datamodule=datamodule)
        print('Start inference test in training stage 1...')
        trainer.predict(model=lightningmodule, dataloaders=dataloader, return_predictions=False)

    # Stage 2
    print('Initializing training stage 2...')

    lightningmodule.advance_stage()
    datamodule._STAGE = 2
    metric_callback_boundary = build_metric_callback(config.TRAIN_STAGE2.METRICS.BOUNDARY)
    metric_callback_region = build_metric_callback(config.TRAIN_STAGE2.METRICS.REGION)
    prediction_writer_stage2 = build_prediction_writer(config.PREDICT.WRITER, mode='train')
    callbacks_s2 = [metric_callback_boundary, metric_callback_region, prediction_writer_stage2]

    trainer, train_hparams = build_trainer(
        train_node=config.TRAIN_STAGE2,
        model_node=config.MODEL,
        env_node=config.ENVIRONMENT,
        extra_callbacks=callbacks_s2
    )
    checkpoint_path = train_hparams.get('checkpoint', None)
    if checkpoint_path is not None:
        print('Loading model parameters from checkpoint: {ckpt}'.format(ckpt=checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        lightningmodule.load_state_dict(checkpoint['state_dict'])
    hparams_stage2.update(train_hparams)

    if config.TRAIN_STAGE2.USE_BATCHSIZE_FINDER:
        if tuner is None:
            import os
            from lightning.pytorch.tuner import Tuner
            from lightning import Trainer
            temp_trainer = Trainer(
                accelerator=config.TRAIN_STAGE2.TRAINER.accelerator, precision=config.TRAIN_STAGE2.TRAINER.precision,
                default_root_dir=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, 'tuner'),
            )
            tuner = Tuner(temp_trainer)
        batch_size = tuner.scale_batch_size(lightningmodule, datamodule=datamodule, mode='binsearch')
    else:
        batch_size = config.TRAIN_STAGE2.BATCH_SIZE
        datamodule.batch_size = batch_size
    hparams_stage2['batch_size'] = batch_size
    trainer.logger.log_hyperparams(hparams_stage2)

    if config.TRAIN_STAGE2.TRAINER.max_epochs > 0:
        print('Start fitting procedure in training stage 2...')
        trainer.fit(model=lightningmodule, datamodule=datamodule)
        print('Start test procedure in training stage 2...')
        trainer.test(model=lightningmodule, datamodule=datamodule, ckpt_path='best')
    else:
        print('Start test procedure in training stage 2...')
        trainer.test(model=lightningmodule, datamodule=datamodule)
    print('Start inference test in training stage 2...')
    trainer.predict(model=lightningmodule, dataloaders=dataloader, return_predictions=False)


if __name__ == '__main__':
    args, config = parse_option()
    main(config, args.stage, args.ignore_warnings)

# EOF
