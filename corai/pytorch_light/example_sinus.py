import os
import time

import numpy as np
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import corai
import corai_plot.tests.test_displayableplot
from corai.pytorch_light.classes_sinus import Sinus_model, MyDataModule, LitProgressBar

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = 0
BATCH_SIZE = 200

seed_everything(42, workers=True)


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
hidden_sizes = [4, 8, 4]
output_size = 1
biases = [True, True, True, True]
activation_functions = [torch.tanh, torch.tanh, torch.relu]
dropout = 0.
epochs = 7500

# Init our model
sinus_model = Sinus_model(input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                          lr=0.01, weight_decay=0.0000001)

# Init the Early Stopper https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.early_stopping.html#pytorch_lightning.callbacks.early_stopping.EarlyStopping
early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=100, verbose=False, mode="min", )

trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=epochs,
                  # progress_bar_refresh_rate=50, # Ignored when a custom progress bar is passed to callbacks.
                  # progress bar over the batches, but is deprecated needs to find alternative.
                  log_every_n_steps=1,
                  callbacks=[early_stop_val_loss, LitProgressBar(refresh_rate=10),
                             # ModelCheckpoint(monitor="val_loss", mode="max", verbose=False) # it is wrong. Without it, works.
                             ])
sinus_data = MyDataModule(xx, yy)

start_time = time.perf_counter()
trainer.fit(sinus_model, datamodule=sinus_data)
print("Total time training: ", time.perf_counter() - start_time, " seconds.")

print(trainer.test(model=sinus_model, ckpt_path="best", dataloaders=sinus_data))

corai.nn_plot_prediction_vs_true(net=sinus_model, plot_xx=plot_xx,
                                 plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy,
                                 device=device)

corai_plot.APlot.show_plot()
