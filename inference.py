import os
import argparse
import warnings
from pathlib import Path
import torch
from lightning import Trainer
from config import get_config
from lightning_module import build_lightning_module
from data import build_inference_dataloader
from lightning_module.proxy import build_prediction_writer


def parse_option():
    parser = argparse.ArgumentParser('MuckSeg inference script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    group.add_argument('--run-folder-path', type=str,
                        help="path to folder which saves model configuration and checkpoint")

    parser.add_argument('--inference-data-path', type=str, required=True, help='path to images to be inference')

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--inference-batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--parameter-path', type=str, help='path to model parameters(checkpoint)')
    parser.add_argument('--inference-result-path', type=str, help='path to output results')
    parser.add_argument('--disable-legend', action='store_true', help='whether to disable legend')
    parser.add_argument('--disable-graindist', action='store_true', help='whether to disable grain distribution graph')
    parser.add_argument('--disable-statistics', action='store_true', help='whether to disable statistic graphs')
    parser.add_argument('--landscape-mode', action='store_true', help='whether to use landscape mode')
    parser.add_argument('--export-annotations', action='store_true', help='whether to export annotations')
    parser.add_argument('--ignore-warnings', action='store_true',
                        help="whether to ignore warnings during execution")

    args, unparsed = parser.parse_known_args()

    arg_mapper = {
        'inference_data_path': {'PREDICT.DATA_PATH': None},
        'inference_batch_size': {'PREDICT.DATAMODULE.batch_size': None},
        'parameter_path': {'PREDICT.CKPT_PATH': None},
        'inference_result_path': {'PREDICT.WRITER.output_dir': None,
                                  'PREDICT.POST_PROCESSING.WRITER.output_dir': None},
        'disable_legend': {'PREDICT.POST_PROCESSING.VISUALIZATION.draw_legend': False if args.disable_legend else True},
        'disable_graindist': {'PREDICT.POST_PROCESSING.VISUALIZATION.draw_graindist': False if args.disable_graindist else True},
        'disable_statistics': {'PREDICT.POST_PROCESSING.VISUALIZATION.draw_statistics': False if args.disable_statistics else True},
        'landscape_mode': {'PREDICT.POST_PROCESSING.VISUALIZATION.landscape_mode': None},
        'export_annotations': {'PREDICT.POST_PROCESSING.EXPORT.export_annotations': None},
    }

    if args.inference_result_path is None:
        if args.run_folder_path is not None:
            args.inference_result_path = (
                    (Path(args.inference_data_path) / 'predictions') / Path(args.run_folder_path).stem).as_posix()
        elif args.cfg is not None:
            args.inference_result_path = (
                        (Path(args.inference_data_path) / 'predictions') / Path(args.cfg).stem).as_posix()

    if args.run_folder_path is not None:
        import yaml
        with open('configs/environment._local.yaml', 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            env_settings = [
                    'ENVIRONMENT.DATA_BASE_PATH',
                    yaml_cfg['ENVIRONMENT']['DATA_BASE_PATH'],
                    'ENVIRONMENT.RESULT_BASE_PATH',
                    yaml_cfg['ENVIRONMENT']['RESULT_BASE_PATH']
                ]
            if args.opts is None:
                args.opts = env_settings
            else:
                args.opts.extend(env_settings)

        try:
            run_path = Path(args.run_folder_path)
            args.cfg = next(iter(run_path.rglob('*Recent.yaml'))).as_posix()
            checkpoint_fld = next(iter(run_path.rglob('checkpoints')))
            _checkpoint_found = False
            for checkpoint in checkpoint_fld.glob('*'):
                with open(checkpoint / 'aliases.txt', 'r') as f:
                    aliases = f.readline()
                if 'best' in aliases:
                    checkpoint_path = next(iter(checkpoint.glob('*.ckpt'))).as_posix()
                    args.parameter_path = checkpoint_path
                    _checkpoint_found = True
            if not _checkpoint_found:
                raise FileNotFoundError('Cannot find model checkpoint')
        except Exception as e:
            warnings.warn('Error occurred while loading settings from run folder:\n{msg}'.format(msg=e.__repr__()))
            if args.cfg is None:
                raise FileNotFoundError('Cannot load configuration file')

    config = get_config(args, arg_mapper)

    return args, config


def main(config, ignore_warnings=False):
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    dataloader = build_inference_dataloader(config)
    prediction_writer = build_prediction_writer(config.PREDICT.WRITER, mode='inference')
    if config.TRAIN_STAGE2.TRAINER.precision in ['16-mixed', 'bf16-mixed', '16', 'bf16', 16]:
        torch.set_float32_matmul_precision('high')
    trainer = Trainer(
        accelerator=config.TRAIN_STAGE2.TRAINER.accelerator, precision=config.TRAIN_STAGE2.TRAINER.precision,
        devices=config.TRAIN_STAGE2.TRAINER.devices, strategy=config.TRAIN_STAGE2.TRAINER.strategy,
        default_root_dir=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, 'predict'), callbacks=[prediction_writer],
        logger=None
    )
    lightningmodule, _, __ = build_lightning_module(config)
    lightningmodule.featuremap_visualizer = None
    lightningmodule.advance_stage()

    trainer.predict(model=lightningmodule, dataloaders=dataloader, return_predictions=False,
                    ckpt_path=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, config.PREDICT.CKPT_PATH))


if __name__ == '__main__':
    args, config = parse_option()
    main(config, args.ignore_warnings)

# EOF