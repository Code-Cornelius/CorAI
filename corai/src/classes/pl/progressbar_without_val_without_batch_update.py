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
