# example from: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# adds on from https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

import os
import time

import numpy as np
import torch
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn

import corai
import corai_plot.tests.test_displayableplot
from config import ROOT_DIR
from corai import decorator_train_disable_no_grad
from corai.pytorch_light.history_dict import History_dict
from corai.pytorch_light.progressbar_without_val_without_batch_update import \
    Progressbar_without_val_without_batch_update

AVAIL_GPUS = 0
BATCH_SIZE = 200000

OUT_PATH = os.path.join(ROOT_DIR, 'corai', 'pytorch_light', 'out')
seed_everything(42, workers=True)


# section ######################################################################
#  #############################################################################
#  classes


class Sinus_model(LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                 lr, weight_decay, aplot_flag=False):
        super().__init__()
        self.model = corai.factory_parametrised_FC_NN(param_input_size=input_size,
                                                      param_list_hidden_sizes=hidden_sizes,
                                                      param_output_size=output_size, param_list_biases=biases,
                                                      param_activation_functions=activation_functions,
                                                      param_dropout=dropout,
                                                      param_predict_fct=None)()

        # By default, every parameter of the __init__ method will be
        # considered a hyper-parameter to the LightningModule
        self.save_hyperparameters(ignore=["input_size", "aplot_flag"])
        self.criterion = nn.MSELoss(reduction='mean')
        self.lr = lr
        self.weight_decay = weight_decay

        if aplot_flag:
            self.aplot = corai_plot.APlot(how=(1, 1))
        else:
            self.aplot = None

    def forward(self, x):
        return self.model(x)

    @decorator_train_disable_no_grad
    def nn_predict_ans2cpu(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log(name="train_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb, log=True):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        if log:
            self.log(name="val_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)

        # plot the prediction, dynamical evolution through epochs
        x_sort, order = torch.sort(x.view(-1))  # sort values that are randomly ordered
        self.plot_prediction(x_sort, y.view(-1)[order], y_hat.view(-1)[order])

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, log=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def plot_prediction(self, inputs, true, prediction):
        if self.aplot is not None:
            self.aplot._axs[0].clear()
            # plot prediction
            self.aplot.uni_plot(0, inputs, true,
                                dict_plot_param={'color': None, 'linestyle': '-', 'linewidth': 1.0, 'markersize': 0.,
                                                 'label': 'True Values'})
            self.aplot.uni_plot(0, inputs, prediction,
                                dict_plot_param={'color': None, 'linestyle': '--', 'linewidth': 2., 'markersize': 0.,
                                                 'label': 'Prediction After training'},
                                dict_ax={'title': "Prediction over Validation Set", 'xlabel': 'Time Axis',
                                         'ylabel': 'Value'})
            self.aplot.show_legend()
            self.aplot.show_and_continue()


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

    def train_dataloader(self):
        return corai.FastTensorDataLoader(self.train_in, self.train_out, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return corai.FastTensorDataLoader(self.val_in, self.val_out, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return corai.FastTensorDataLoader(self.input, self.output, batch_size=BATCH_SIZE)


# section ######################################################################
#  #############################################################################
#  code

# Define the exact solution
def exact_solution(x):
    return torch.sin(x)

############################## GLOBAL PARAMETERS
n_samples = 2000  # Number of training samples
sigma = 0.01  # Noise level
device = corai.pytorch_device_setting('cpu')
SILENT = False
early_stop_train = corai.Early_stopper_training(patience=20, silent=SILENT, delta=-1E-6)
early_stop_valid = corai.Early_stopper_validation(patience=20, silent=SILENT, delta=-1E-6)
early_stoppers = (early_stop_train, early_stop_valid)
############################# DATA CREATION
# exact grid
plot_xx = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
plot_yy = exact_solution(plot_xx).reshape(-1, )
plot_yy_noisy = (exact_solution(plot_xx) + sigma * torch.randn(plot_xx.shape)).reshape(-1, )

# random points for training
xx = 2 * np.pi * torch.rand((n_samples, 1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

input_size = 1
hidden_sizes = [32, 64, 32]
output_size = 1
biases = [True, True, True, True]
activation_functions = [torch.tanh, torch.tanh, torch.relu]
dropout = 0.
epochs = 5

############################### Init our model
sinus_model = Sinus_model(input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                          lr=0.1, weight_decay=0.0000001, aplot_flag=True)
############################### Init the Early Stopper
period_log =1
early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=100 // period_log, verbose=False,
                                    mode="min", )

logger = CSVLogger(os.path.join(OUT_PATH, 'csv_logs'))
logger_custom = History_dict(aplot_flag=True, frequency_epoch_logging=period_log)
chckpnt = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False, save_top_k=3)

trainer = Trainer(
    default_root_dir=OUT_PATH,
    gpus=AVAIL_GPUS, max_epochs=epochs,
    logger=[logger, logger_custom],
    check_val_every_n_epoch=period_log,
    num_sanity_val_steps=0,
    callbacks=[early_stop_val_loss, Progressbar_without_val_without_batch_update(refresh_rate=10),
               chckpnt, ])
sinus_data = MyDataModule(xx, yy)

start_time = time.perf_counter()
trainer.fit(sinus_model, datamodule=sinus_data)
final_time = time.perf_counter() - start_time
#TODO add this final time to chktpt?
print("Total time training: ", time.perf_counter() - start_time, " seconds.")

print(trainer.test(model=sinus_model, ckpt_path="best", dataloaders=sinus_data))

corai.nn_plot_prediction_vs_true(net=sinus_model, plot_xx=plot_xx,
                                 plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy,
                                 device=device)

# todo issue with the name of columns
estimator_history = logger_custom.to_estim_history(checkpoint=chckpnt)
estimator_history.to_json(os.path.join(OUT_PATH, 'estims', 'estim1.json'), compress=False)
history_plot = corai.Relplot_history(estimator_history)
history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot=None, log_axis_for_loss=True)
history_plot.lineplot(log_axis_for_loss=True)

corai.nn_plot_prediction_vs_true(net=sinus_model, plot_xx=plot_xx,
                                 plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy,
                                 device=device)
corai_plot.APlot.show_plot()

# todo path, maybe we could use the functions we wrote  `factory_fct_linked_path`

# TODO UNWANTED OUTPUT IN CONSOL, WHERE FROM?
#  Total time training:  2.2748919  seconds.
#  Testing: 100%|██████████| 1/1 [00:00<00:00,  4.85it/s]--------------------------------------------------------------------------------
#  DATALOADER:0 TEST RESULTS
#  {}
#  --------------------------------------------------------------------------------
#  Testing: 100%|██████████| 1/1 [00:00<00:00,  4.83it/s]
#  [{}]