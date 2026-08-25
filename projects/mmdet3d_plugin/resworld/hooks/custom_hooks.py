from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
from mmcv.parallel import is_module_wrapper


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)

    def before_train_iter(self, runner):
        """Inject the runner iter count into the head (gate warmup).

        The head's ``iter`` attribute drives the risk-gated injection
        warmup (``gate_warmup_iters``, design doc Stage 2): before_train_iter
        runs right before the forward, so the gate sees the exact iter
        count of the upcoming step.
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        head = getattr(model, 'pts_bbox_head', None)
        if head is not None:
            head.iter = runner.iter


@HOOKS.register_module()
class DiagLoggerHook(Hook):
    """Print the head's cached [DIAG] line on the TextLoggerHook cadence.

    Uses the same every_n_iters(runner, interval) gate as TextLoggerHook
    (log_config.interval in resworld_config.py), so the diagnostic aligns
    with the per-100-iter loss output instead of drifting across epochs
    (3517 iters/epoch does not divide evenly by 100). Only rank 0 prints,
    via runner.logger so the line lands in the same training log.
    """

    def __init__(self, interval=100):
        self.interval = interval

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        if runner.rank != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        msg = getattr(model.pts_bbox_head, '_diag_msg', None)
        if msg:
            runner.logger.info(msg)

