import sys

from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, convert_inf, TQDMProgressBar
from tqdm import tqdm


class Progressbar_without_val_without_batch_update(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_validation_tqdm(self):
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(disable=True)
        return bar

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(desc="Training", position=(2 * self.process_position), disable=self.is_disabled,
                   leave=True, dynamic_ncols=True, file=sys.stdout, colour='blue')
        return bar

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        total_batches = self.total_train_batches + self.total_val_batches
        total_batches = convert_inf(total_batches)
        if self._should_update(self.train_batch_idx, total_batches):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
