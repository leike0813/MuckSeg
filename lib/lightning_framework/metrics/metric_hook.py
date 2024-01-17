import inspect
from enum import IntEnum, IntFlag
from torch import nn
import torchmetrics as metrics


class EvaluationStage(IntEnum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class EvaluationLevel(IntFlag):
    BATCH = 1
    EPOCH = 2


class MetricHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.METRIC_HOOK_TRAIN = nn.ModuleList()
        self.METRIC_HOOK_VALID = nn.ModuleList()
        self.METRIC_HOOK_TEST = nn.ModuleList()
        self.METRIC_LIB = {}

    def register_metrics(self, node):
        category = node.get('CATEGORY', 'default')
        append_category_name = node.get('APPEND_CATEGORY_NAME', False)
        for stage, stage_node in node.items():
            if stage == 'TRAINING':
                _stage = EvaluationStage.TRAINING
            elif stage == 'VALIDATION':
                _stage = EvaluationStage.VALIDATION
            elif stage == 'TEST':
                _stage = EvaluationStage.TEST
            else:
                continue

            for metric_type, params in stage_node.items():
                level = EvaluationLevel(params.pop('LEVEL', 2))
                alias = params.pop('ALIAS', None)

                self.register_metric(category, metric_type, _stage, level, params, alias=alias, append_category_name=append_category_name)

    def register_metric(self, category, metric_type, stage, level, params, alias=None, append_category_name=False):
        stage = EvaluationStage(stage)
        if not isinstance(level, (int, EvaluationLevel)) or level < 1 or level > 3:
            raise ValueError('level must be valid metrics.EvaluationLevel flag or integer')

        if stage == EvaluationStage.TRAINING:
            _stage = 'train'
            hook = self.METRIC_HOOK_TRAIN
        elif stage == EvaluationStage.VALIDATION:
            _stage = 'valid'
            hook = self.METRIC_HOOK_VALID
        elif stage == EvaluationStage.TEST:
            _stage = 'test'
            hook = self.METRIC_HOOK_TEST

        _levels = []
        if EvaluationLevel.BATCH in level:
            _levels.append('batch')
        if EvaluationLevel.EPOCH in level:
            _levels.append('epoch')

        metric_generator = getattr(metrics, metric_type)
        metric_func = metric_generator(**params)
        cur_idx = len(hook)
        hook.append(metric_func)
        for lvl in _levels:
            lib = self.METRIC_LIB.setdefault(category, {}).setdefault('{stg}_{lvl}'.format(stg=_stage, lvl=lvl), {})
            reg_name = '{stg}/{lvl}/{nme}{cat}'.format(
                stg=_stage,
                lvl=lvl,
                nme=alias if alias else metric_type,
                cat='_' + category if append_category_name else ''
            )
            lib[reg_name] = cur_idx

    def inspect_caller(self, caller, category):
        if caller == 'training_step':
            hooks = self.METRIC_HOOK_TRAIN
            lib = self.METRIC_LIB[category].get('train_batch', {})
            lib_calc_only = self.METRIC_LIB[category].get('train_epoch', {})
        elif caller == 'on_train_epoch_end':
            hooks = self.METRIC_HOOK_TRAIN
            lib = self.METRIC_LIB[category].get('train_epoch', {})
            lib_calc_only = {}
        elif caller == 'validation_step':
            hooks = self.METRIC_HOOK_VALID
            lib = self.METRIC_LIB[category].get('valid_batch', {})
            lib_calc_only = self.METRIC_LIB[category].get('valid_epoch', {})
        elif caller == 'on_validation_epoch_end' or caller == 'on_validation_end':
            hooks = self.METRIC_HOOK_VALID
            lib = self.METRIC_LIB[category].get('valid_epoch', {})
            lib_calc_only = {}
        elif caller == 'test_step':
            hooks = self.METRIC_HOOK_TEST
            lib = self.METRIC_LIB[category].get('test_batch', {})
            lib_calc_only = self.METRIC_LIB[category].get('test_epoch', {})
        elif caller == 'on_test_epoch_end' or caller == 'on_test_end':
            hooks = self.METRIC_HOOK_TEST
            lib = self.METRIC_LIB[category].get('test_epoch', {})
            lib_calc_only = {}
        else:
            raise RuntimeError('"call_metrics" called by invalid hook')

        return hooks, lib, lib_calc_only

    def calc_metrics(self, preds=None, target=None, category='default', log=True):
        caller = inspect.stack()[1][3]
        hooks, lib, lib_calc_only = self.inspect_caller(caller, category)
        ret = {}
        if preds is not None and target is not None:
            for metric_name, hook_idx in lib_calc_only.items():
                if hook_idx not in lib.values(): # only calculate epoch-only metrics
                    _ = hooks[hook_idx](preds, target)
            for metric_name, hook_idx in lib.items():
                metric_value = hooks[hook_idx](preds, target)
                if log:
                    if caller in ['on_validation_end', 'on_test_end']:
                        self.logger.experiment.log_metric(self.logger._run_id, metric_name, metric_value)
                    else:
                        self.log(metric_name, metric_value)
                ret[metric_name] = metric_value
        elif preds is None and target is None:
            for metric_name, hook_idx in lib.items():
                metric_value = hooks[hook_idx].compute()
                hooks[hook_idx].reset()
                if log:
                    if caller in ['on_validation_end', 'on_test_end']:
                        self.logger.experiment.log_metric(self.logger._run_id, metric_name, metric_value)
                    else:
                        self.log(metric_name, metric_value)
                ret[metric_name] = metric_value
        else:
            raise ValueError('Invalid inputs, preds and target must either be None or Tensor simutaniously')

        return ret

# EOF