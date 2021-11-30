# example from: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# adds on from https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
import os
import sys

import torch
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, convert_inf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import corai
from corai import decorator_train_disable_no_grad

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = 0
BATCH_SIZE = 200


# Keep in Mind - A LightningModule is a PyTorch nn.Module - it just has a few more helpful features.
class Sinus_model(LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                 lr, weight_decay):
        super().__init__()
        self.model = corai.factory_parametrised_FC_NN(param_input_size=input_size,
                                                      param_list_hidden_sizes=hidden_sizes,
                                                      param_output_size=output_size, param_list_biases=biases,
                                                      param_activation_functions=activation_functions,
                                                      param_dropout=dropout,
                                                      param_predict_fct=None)()

        # By default, every parameter of the __init__ method will be
        # considered a hyperparameter to the LightningModule
        self.save_hyperparameters(ignore=["input_size"])
        self.criterion = nn.MSELoss(reduction='sum')
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    @decorator_train_disable_no_grad
    def nn_predict_ans2cpu(self,x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log(name="train_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)
        # wip add another loss
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log(name="val_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        # x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # self.log("test_loss", loss)
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class MyDataModule(LightningDataModule):
    def __init__(self, x, y):
        super().__init__()
        self.input = x
        self.output = y

    def setup(self, stage=None):
        training_size = int(90. / 100. * len(self.input))
        self.train_in = self.input[:training_size]
        self.train_out = self.output[:training_size]
        self.val_in = self.input[training_size:]
        self.val_out = self.output[training_size:]


    # train_dataloader(), val_dataloader(), and test_dataloader() all return PyTorch DataLoader instances
    # that are created by wrapping their respective datasets that we prepared in setup()
    def train_dataloader(self):
        # return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)
        return corai.FastTensorDataLoader(self.train_in, self.train_out,
                                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)
        return corai.FastTensorDataLoader(self.val_in, self.val_out,
                                          batch_size=BATCH_SIZE)


    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)
        return corai.FastTensorDataLoader(self.input, self.output,
                                          batch_size=BATCH_SIZE)


# section ######################################################################
#  #############################################################################
#  progress bar:


class LitProgressBar(ProgressBar):
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

