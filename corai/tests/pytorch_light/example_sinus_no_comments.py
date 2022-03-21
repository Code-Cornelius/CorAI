# example from: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# adds on from https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
import time

import numpy as np
import torch
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn

import corai
import corai_plot.tests.test_displayableplot
from config import ROOT_DIR
from corai import decorator_train_disable_no_grad, Estim_history
from corai.src.classes.pl.history_dict import History_dict
from corai.src.classes.pl.progressbar_without_val_batch_update import \
    Progressbar_without_val_batch_update
from corai.tests.sinus_dataset_generator import data_sinus
from corai_util.tools.src.function_file import remove_file
from corai_util.tools.src.function_writer import factory_fct_linked_path

path_linker = factory_fct_linked_path(ROOT_DIR, 'corai/tests/pytorch_light/')
history_path = path_linker(['out', 'estim', 'estim_1.json'])
model_path = path_linker(['out', 'model', ''])
model_name = 'sinus_model'
# if set to True, the model will be overwritten, otherwise the new model will be saved with 'model_name-vx'
OVERWRITE = True
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

        self.save_hyperparameters()
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

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log(name="val_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)

        # plot the prediction, dynamical evolution through epochs
        x_sort, order = torch.sort(x.view(-1))  # sort values that are randomly ordered
        self.plot_prediction(x_sort, y.view(-1)[order], y_hat.view(-1)[order])

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # will print to the console
        self.log(name="test_loss", value=loss, prog_bar=True, on_step=False, on_epoch=True)

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
                                dict_ax={'title': "Dynamical Image of Prediction over Validation Set",
                                         'xlabel': 'Time Axis',
                                         'ylabel': 'Value'})
            self.aplot.show_legend()
            self.aplot.show_and_continue()


class MyDataModule(LightningDataModule):
    def __init__(self, x, y, batch_size):
        super().__init__()
        self.input = x
        self.output = y
        self.batch_size = batch_size

    def setup(self, stage=None):
        training_size = int(90. / 100. * len(self.input))
        self.train_in = self.input[:training_size]
        self.train_out = self.output[:training_size]
        self.val_in = self.input[training_size:]
        self.val_out = self.output[training_size:]

    def train_dataloader(self):
        return corai.FastTensorDataLoader(self.train_in, self.train_out, batch_size=self.batch_size)

    def val_dataloader(self):
        return corai.FastTensorDataLoader(self.val_in, self.val_out, batch_size=self.batch_size)

    def test_dataloader(self):
        return corai.FastTensorDataLoader(self.input, self.output, batch_size=self.batch_size)


if __name__ == '__main__':
    ############################# DATA CREATION
    train_X, train_Y, testing_X, testing_Y, plot_xx, plot_yy, plot_yy_noisy, xx, yy = data_sinus()

    ############################# parameters for model

    input_size = 1
    hidden_sizes = [4, 8, 4]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.tanh]
    dropout = 0.
    epochs = 5000
    AVAIL_GPUS = 0
    BATCH_SIZE = 200000

    ############################### Init our model
    sinus_model = Sinus_model(input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                              lr=0.01, weight_decay=0.00001, aplot_flag=True)
    ############################### Init the Early Stopper
    period_log = 20
    early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=1E-3, patience=100 // period_log,
                                        verbose=False, mode="min")

    logger_custom = History_dict(metrics=["val_loss", "train_loss"], aplot_flag=True,
                                 frequency_epoch_logging=period_log, )

    chckpnt = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False, save_top_k=1,
                              dirpath=path_linker(['out', 'model']), filename=model_name)

    # If a model already exists and the overwrite flag is on, delete the preivious model to present pl to save a new
    # version
    if OVERWRITE:
        remove_file(model_path + model_name + '.ckpt')

    trainer = Trainer(default_root_dir=path_linker(['out']),
                      gpus=AVAIL_GPUS, max_epochs=epochs,
                      logger=[logger_custom],
                      check_val_every_n_epoch=period_log,
                      num_sanity_val_steps=0,
                      callbacks=[early_stop_val_loss, Progressbar_without_val_batch_update(refresh_rate=10),
                                 chckpnt,])
    sinus_data = MyDataModule(xx, yy, BATCH_SIZE)

    # section ######################################################################
    #  #############################################################################
    #  Training

    start_time = time.perf_counter()
    trainer.fit(sinus_model, datamodule=sinus_data)
    final_time = time.perf_counter() - start_time
    train_time = np.round(time.perf_counter() - start_time, 2)
    print("Total time training: ", train_time, " seconds. In average, it took: ",
          np.round(train_time / trainer.current_epoch, 4), " seconds per epochs.")
    estimator_history = logger_custom.to_estim_history(checkpoint=chckpnt, train_time=final_time)
    estimator_history.to_json(history_path, compress=False)
    trainer.test(model=sinus_model, ckpt_path="best", dataloaders=sinus_data)

    # section ######################################################################
    #  #############################################################################
    #  plot

    corai.nn_plot_prediction_vs_true(net=sinus_model, plot_xx=plot_xx, plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy)
    history_plot = corai.Relplot_history(estimator_history)
    history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot=None, log_axis_for_loss=True)
    history_plot.lineplot(log_axis_for_loss=True)

    # section ######################################################################
    #  #############################################################################
    #  Loading the model back

    # TODO WE WANT TO BE ABLE TO CHOSE WHERE THE CHECKPT IS LOCATED. IN EXAMPLE_HP and in the SINUS_EXAMPLE.
    model = Sinus_model.load_from_checkpoint(chckpnt.best_model_path)
    # OR
    model = Sinus_model.load_from_checkpoint(model_path + model_name + '.ckpt')
    estim = Estim_history.from_json(history_path, compressed=False)

    corai_plot.APlot.show_plot()
