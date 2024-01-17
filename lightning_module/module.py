# coding: utf-8
import os
from pathlib import Path
import shutil
import lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from utils.prediction_cutter import PredictionCutter
from lib.lightning_framework.callbacks import PredictionWriter
from utils import map_sparse_to_dense, extract_encoder_state_dict

class MuckSeg_Lightning_Module(L.LightningModule):
    """MuckSeg model wrapper for lightning functionality

    Args:
        model (lightning.LightningModule): MuckSeg model.
        loss_fn_stage1 (callable): A function for calculating the loss in training stage 1.
        loss_fn_stage2 (callable): A function for calculating the loss in training stage 2.
        loss_fn_finetune (callable): A function for calculating the loss in fine-tuning.
        optimizer_stage1 (torch.optim.optimizer.Optimizer): Optimizer instance for training stage 1.
        optimizer_stage2 (torch.optim.optimizer.Optimizer): Optimizer instance for training stage 2 and fine-tuning.
        scheduler_stage1 (torch.optim.lr_scheduler.LRScheduler or callable): Learning scheduler instance for training stage 1.
        scheduler_stage2 (torch.optim.lr_scheduler.LRScheduler or callable): Learning scheduler instance for training stage 2 and fine-tuning.
        test_visualizer (TestVisualizer): TestVisualizer instance for visualization of test outputs.
        post_processor (PostProcessingThread): PostProcessingThread instance for post-processing.
        config (yacs.config.CfgNode): Config node instance.
        featuremap_visualizer (FeatureMapVisualizer, optional): FeatureMapVisualizer instance for visualization of featuremaps.
        example_input (torch.Tensor, optional): Example model input for model summarization.
    """
    def __init__(self, model, loss_fn_stage1, loss_fn_stage2, loss_fn_finetune, optimizer_stage1, optimizer_stage2,
                 scheduler_stage1, scheduler_stage2, test_visualizer, post_processor,
                 config, featuremap_visualizer=None, example_input=None):
        super().__init__()
        self.model = model
        self.__STAGE = 1

        self.loss_func_stage1 = loss_fn_stage1
        self.loss_func_stage2 = loss_fn_stage2
        self.loss_func_finetune = loss_fn_finetune
        self.optimizer_stage1 = optimizer_stage1
        self.optimizer_stage2 = optimizer_stage2
        self.scheduler_stage1 = scheduler_stage1
        self.scheduler_stage2 = scheduler_stage2
        self.test_visualizer = test_visualizer
        self.post_processor = post_processor
        self.featuremap_visualizer = featuremap_visualizer
        self.config = config
        self.predict_step_outputs = {}
        self.predict_step_outputs['featuremap_seeds'] = []
        self._omit_orig_image = True
        if example_input is not None:
            self._example_input_array = example_input

    def forward(self, x):
        x, xi = self.model.encoder(x)
        x = self.model.neck(x)
        x, _ = self.model.decoder_stage1(x, xi)
        x, _ = self.model.head_stage1(x, _)
        return x

    def forward_stage2(self, x):
        x, xi = self.model.encoder(x)
        x = self.model.neck(x)
        x = self.model.decoder_stage1(x, xi)
        xb, xr, xb0, xr0 = self.model.decoder_stage2(x, xi)
        xb, xr, _, __ = self.model.head_stage2(xb, xr, xb0, xr0)
        return xb, xr

    def forward_finetune(self, x):
        x, xi = self.model.encoder(x)
        x = self.model.neck(x)
        x = self.model.decoder_stage1(x, xi)
        xb, xr = self.model.decoder_stage2(x, xi)
        xb, xr = self.model.head_stage2(xb, xr)
        return xb, xr

    def configure_optimizers(self):
        if self.__STAGE == 1:
            return {
                'optimizer': self.optimizer_stage1,
                'lr_scheduler': self.scheduler_stage1
            }
        elif self.__STAGE == 2:
            return {
                'optimizer': self.optimizer_stage2,
                'lr_scheduler': self.scheduler_stage2
            }

    def load_fcmae(self, fcmae_ckpt):
        self.model.encoder.load_state_dict(map_sparse_to_dense(extract_encoder_state_dict(fcmae_ckpt)))

    def advance_stage(self):
        assert self.__STAGE == 1, f'The lightning module must be at stage 1 to proceed advancing, but current stage is {self.__STAGE}'
        self.__STAGE = 2
        self.forward = self.forward_stage2
        self.advance_batch = self.advance_batch_stage2
        self.make_predict = self.make_predict_stage2
        self.predict_step_outputs['featuremap_seeds'] = []
        self.model.advance_stage()

    def advance_finetune(self):
        assert self.__STAGE == 2, f'The lightning module must be at stage 2 before fine-tuning, but current stage is {self.__STAGE}'
        self.forward = self.forward_finetune
        self.advance_batch = self.advance_batch_finetune
        self.predict_step_outputs['featuremap_seeds'] = []
        self.model.advance_finetune()

    def advance_batch(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        y = F.interpolate(y, scale_factor=0.25, mode='bilinear')
        loss = self.loss_func_stage1(y_pred, y)
        return {'loss': loss,
                'metric_Raw_Region': (y_pred[0], y.round().int())}

    def advance_batch_stage2(self, batch, batch_idx):
        x, y_bound, y_region = batch
        y_pred = self.model(x)
        loss = self.loss_func_stage2(y_pred, y_bound, y_region)
        return {'loss': loss,
                'metric_Boundary': (y_pred[0], y_bound.round().int()),
                'metric_Region': (y_pred[1], y_region.round().int())}

    def advance_batch_finetune(self, batch, batch_idx):
        x, y_bound, y_region = batch
        y_pred = self.model(x)
        loss = self.loss_func_finetune(y_pred, y_bound, y_region)
        return {'loss': loss,
                'metric_Boundary': (y_pred[0], y_bound.round().int()),
                'metric_Region': (y_pred[1], y_region.round().int())}

    def make_predict(self, orig_imgs):
        orig_imgs = self.prediction_cutter.cut_roi(orig_imgs)

        if self.predict_mode == self.prediction_cutter.PredictMode.FULL_SIZE:
            pred_region = self(orig_imgs)
        elif self.predict_mode == self.prediction_cutter.PredictMode.THREE_FOLD:
            orig_imgs = self.prediction_cutter.cut_fold(orig_imgs)
            pred_region = []
            for i in range(3):
                pred_region.append(TF.crop(
                    self(orig_imgs[i]),
                    int(self.prediction_cutter.cut_coords[i, 2] * 0.25),
                    int(self.prediction_cutter.cut_coords[i, 3] * 0.25),
                    int(self.prediction_cutter.cut_coords[i, 4] * 0.25),
                    int(self.prediction_cutter.cut_coords[i, 5] * 0.25)
                ))
            pred_region = torch.cat(pred_region, dim=2)
        elif self.predict_mode == self.prediction_cutter.PredictMode.PATCH:
            orig_imgs = self.prediction_cutter.cut_patch(orig_imgs)
            pred_region = []
            for i in range(self.patches_shape[0]):
                pred_row_region = []
                for j in range(self.patches_shape[1]):
                    pred_row_region.append(TF.crop(
                        self(orig_imgs[i][j]),
                        int(self.prediction_cutter.cut_coords[i, j, 2] * 0.25),
                        int(self.prediction_cutter.cut_coords[i, j, 3] * 0.25),
                        int(self.prediction_cutter.cut_coords[i, j, 4] * 0.25),
                        int(self.prediction_cutter.cut_coords[i, j, 5] * 0.25),
                    ))
                pred_region.append(torch.cat(pred_row_region, dim=3))
            pred_region = torch.cat(pred_region, dim=2)

        pred_region.sigmoid_()
        pred_region = self.prediction_cutter.concatenate(pred_region, scale_factor=0.25)

        return {'region.png': pred_region}

    def make_predict_stage2(self, orig_imgs):
        orig_imgs = self.prediction_cutter.cut_roi(orig_imgs)

        if self.predict_mode == self.prediction_cutter.PredictMode.FULL_SIZE:
            pred_boundary, pred_region = self(orig_imgs)
        elif self.predict_mode == self.prediction_cutter.PredictMode.THREE_FOLD:
            orig_imgs = self.prediction_cutter.cut_fold(orig_imgs)
            pred_boundary = []
            pred_region = []
            for i in range(3):
                _ = self(orig_imgs[i])
                pred_boundary.append(TF.crop(_[0], *self.prediction_cutter.cut_coords[i, 2:]))
                pred_region.append(TF.crop(_[1], *self.prediction_cutter.cut_coords[i, 2:]))
            pred_boundary = torch.cat(pred_boundary, dim=2)
            pred_region = torch.cat(pred_region, dim=2)
        elif self.predict_mode == self.prediction_cutter.PredictMode.PATCH:
            orig_imgs = self.prediction_cutter.cut_patch(orig_imgs)
            pred_boundary = []
            pred_region = []
            for i in range(self.patches_shape[0]):
                pred_row_boundary = []
                pred_row_region = []
                for j in range(self.patches_shape[1]):
                    _ = self(orig_imgs[i][j])
                    pred_row_boundary.append(TF.crop(_[0], *self.prediction_cutter.cut_coords[i, j, 2:]))
                    pred_row_region.append(TF.crop(_[1], *self.prediction_cutter.cut_coords[i, j, 2:]))
                pred_boundary.append(torch.cat(pred_row_boundary, dim=3))
                pred_region.append(torch.cat(pred_row_region, dim=3))
            pred_boundary = torch.cat(pred_boundary, dim=2)
            pred_region = torch.cat(pred_region, dim=2)

        pred_boundary.sigmoid_()
        pred_region.sigmoid_()
        pred_boundary = self.prediction_cutter.concatenate(pred_boundary)
        pred_region = self.prediction_cutter.concatenate(pred_region)

        return {'boundary.png': pred_boundary, 'region.png': pred_region}

    def get_run_basefld(self):
        if self.trainer.checkpoint_callback:
            _run_basefld = Path(self.trainer.checkpoint_callback.dirpath).parent
            if self.mlflow_logger_available:
                return _run_basefld if _run_basefld.name == self.logger.run_id else None
            else:
                return _run_basefld if _run_basefld == Path(self.logger.log_dir) else None
        else:
            return (Path(self.trainer.default_root_dir) / self.logger.experiment_id) / self.logger.run_id \
                if self.mlflow_logger_available else Path(self.trainer.logger.log_dir)

    def setup(self, stage):
        self.mlflow_logger_available = isinstance(self.logger, MLFlowLogger)
        if stage in ['fit', 'test']:
            self.run_basefld = self.get_run_basefld()
            self._omit_orig_image = False
            if self.run_basefld:
                self.result_img_fld = self.run_basefld / 'result_img'
                self.featuremap_fld = self.run_basefld / 'feature_map'
                self.inference_example_fld = self.run_basefld / 'inference_example'

                self.test_visualizer.basefld = self.result_img_fld
                if self.featuremap_visualizer is not None:
                    self.featuremap_visualizer.basefld = self.featuremap_fld

                self.post_processor.prediction_writer.output_dir = self.inference_example_fld

    def teardown(self, stage):
        if stage == 'test':
            if self.run_basefld and (self.run_basefld / 'checkpoints').exists():
                shutil.rmtree(self.run_basefld / 'checkpoints')
            if hasattr(self, 'result_img_fld') and self.result_img_fld.exists():
                shutil.rmtree(self.result_img_fld)
        if stage == 'predict':
            if hasattr(self, 'inference_example_fld') and self.inference_example_fld.exists():
                shutil.rmtree(self.inference_example_fld)
            if hasattr(self, 'featuremap_fld') and self.featuremap_fld.exists():
                shutil.rmtree(self.featuremap_fld)

    def on_fit_start(self):
        config_filename = '{type}_{spec}_Recent.yaml'.format(type=self.config.MODEL.TYPE, spec=self.config.MODEL.SPEC_NAME)
        config_out_file = os.path.join(
            os.path.join(self.config.ENVIRONMENT.PROJECT_PATH, self.config.CONFIG_OUTPUT_PATH),
            config_filename)
        with open(config_out_file, "w") as f:
            f.write(self.config.dump_visible())
        if self.mlflow_logger_available:
            self.logger.experiment.log_artifact(self.logger._run_id, config_out_file, f'config')

        if self.config.FULL_DUMP:
            config_filename_full = '{type}_{spec}_Full.yaml'.format(type=self.config.MODEL.TYPE, spec=self.config.MODEL.SPEC_NAME)
            config_full_out_file = os.path.join(
                os.path.join(self.config.ENVIRONMENT.PROJECT_PATH, self.config.CONFIG_OUTPUT_PATH),
                config_filename_full)
            with open(config_full_out_file, "w") as f:
                f.write(self.config.dump())
            if self.mlflow_logger_available:
                self.logger.experiment.log_artifact(self.logger._run_id, config_full_out_file, f'config')

        if self.mlflow_logger_available:
            for path in self.config.MODEL.FILE_PATHS:
                self.logger.experiment.log_artifact(self.logger._run_id, path, f'model')

    def training_step(self, batch, batch_idx):
        return self.advance_batch(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.advance_batch(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.__STAGE == 1:
            x, y = batch
            y_pred = self.model(x)
            y = F.interpolate(y, scale_factor=0.25, mode='bilinear')
            loss = self.loss_func_stage1(y_pred, y)
            return {'loss': loss,
                    'metric_Raw_Region': (y_pred[0], y.round().int())}

        elif self.__STAGE == 2:
            x, y_bound, y_region = batch
            y_pred = self.model(x)
            if self.model.head_stage2.side_output_detached:
                loss = self.loss_func_finetune(y_pred, y_bound, y_region)
            else:
                loss = self.loss_func_stage2(y_pred, y_bound, y_region)

            img_path_bound = self.test_visualizer(x, y_pred[0], y_bound, batch_idx, 'Boundary')
            img_path_region = self.test_visualizer(x, y_pred[1], y_region, batch_idx, 'Region')
            if self.mlflow_logger_available:
                for i in range(len(img_path_bound)):
                    self.logger.experiment.log_artifact(
                        self.logger._run_id,
                        img_path_bound[i],
                        f"result_img"
                    )
                for i in range(len(img_path_region)):
                    self.logger.experiment.log_artifact(
                        self.logger._run_id,
                        img_path_region[i],
                        f"result_img"
                    )

            return {'loss': loss,
                    'metric_Boundary': (y_pred[0], y_bound.round().int()),
                    'metric_Region': (y_pred[1], y_region.round().int())}

    def on_predict_start(self):
        for callback in self.trainer.callbacks:
            if isinstance(callback, PredictionWriter):
                if hasattr(self, 'inference_example_fld'): # Use this as the criterion for distinguish if the trainer is running at the training mode
                    callback.output_dir = self.inference_example_fld
                else:
                    if self.featuremap_visualizer is not None:
                        self.featuremap_visualizer.basefld = callback.output_dir / 'feature_map'

        self.prediction_cutter = PredictionCutter(self, self.config)
        self.predict_mode, self.patches_shape = self.prediction_cutter.find_predict_mode()

        if self.__STAGE == 2:
            self.post_processor.setDaemon(True)
            self.post_processor.start()

    def predict_step(self, batch, batch_idx):
        garbage_collection_cuda()

        orig_imgs, img_names = batch
        orig_imgs_denormalized = self.trainer.predict_dataloaders.dataset.denormalizer(orig_imgs)
        if batch_idx < self.config.VISUALIZATION.NUM_FEATUREMAP_SAMPLES:
            self.predict_step_outputs['featuremap_seeds'].append(
                TF.center_crop(orig_imgs, (self.config.MODEL.IMAGE_SIZE, self.config.MODEL.IMAGE_SIZE))
            )

        predicts = self.make_predict(orig_imgs)

        if self.__STAGE == 1:
            orig_imgs_denormalized = F.interpolate(orig_imgs_denormalized, scale_factor=0.25, mode="bilinear")
        if self.__STAGE == 2:
            for i in range(len(img_names)):
                if self.post_processor.buffer.full():
                    print('buffer is full, waiting for post processing thread')
                self.post_processor.buffer.put((orig_imgs_denormalized[i].cpu(), predicts['boundary.png'][i].cpu(), predicts['region.png'][i].cpu(), img_names[i]))

        ret = {'.orig.jpg' if self._omit_orig_image else 'orig.jpg': orig_imgs_denormalized}
        ret.update(predicts)

        return ret, img_names

    def on_predict_end(self):
        if self.__STAGE == 2:
            print('inference finished, waiting for post processing thread to finish')
            self.post_processor.stop()
            self.post_processor.join()
            print('post processing finished')

            if self.mlflow_logger_available:
                for img_path in self.post_processor.results['image_paths']:
                    self.logger.experiment.log_artifact(
                        self.logger._run_id,
                        img_path,
                        f"inference_example"
                    )
                self.logger.experiment.log_artifact(
                    self.logger._run_id,
                    self.post_processor.results['statistics_file_path'][1],
                    f"inference_example"
                )

        if self.featuremap_visualizer is not None:
            for img_idx, img in enumerate(self.predict_step_outputs['featuremap_seeds']):
                fmaps = self.model.forward_featuremaps(img)
                paths = self.featuremap_visualizer(fmaps, img_idx)
                if self.mlflow_logger_available:
                    for path in paths.values():
                        self.logger.experiment.log_artifact(
                            self.logger._run_id,
                            path,
                            f"feature_map"
                        )

# EOF