# example from: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# adds on from https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
import os
import time

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from corai.src.classes.pl.history_dict import History_dict
from corai.src.classes.pl.progressbar_without_val_batch_update import \
    Progressbar_without_val_batch_update
from corai.tests.pytorch_light.classes_mnist_with_comments import MNISTModel
from corai.tests.pytorch_light.example_sinus_no_comments import MyDataModule

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 256

seed_everything(42, workers=True)

############################### Init our model
mnist_model = MNISTModel(learning_rate=0.001)

############################### Init the Early Stopper
# min_delta: if positive: asks for improvement in direction of mode. If negative, it verifies model does not unlearn.
# https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.early_stopping.html#pytorch_lightning.callbacks.early_stopping.EarlyStopping
# they say absolute different but it is difference.

period_log = 2
# divide the patience bc:
# "It must be noted that the patience parameter counts the number of validation checks
# with no improvement, and not the number of training epochs."
early_stop_val_acc = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=10 // period_log, verbose=False,
                                   mode="max", )
# stopping_threshold = 0.9)
early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10 // period_log, verbose=False,
                                    mode="min", )
early_stop_train_loss = EarlyStopping(monitor="train_loss", min_delta=-1E-3, patience=10 // period_log, verbose=False,
                                      mode="min", )

###############################Init the loggers and checkpoints
logger = CSVLogger("logs")
logger_tf = TensorBoardLogger("./lightning_logs/")
chckpnt = ModelCheckpoint(monitor="val_acc", mode="max", verbose=True)

############################### Initialize a trainer # logger :
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-frequency
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=100,
                  # progress_bar_refresh_rate=50, # Ignored when a custom progress bar is passed to callbacks.
                  # progress bar over the batches, but is deprecated needs to find alternative.
                  logger=[logger, logger_tf,
                          History_dict(aplot_flag=True, frequency_epoch_logging=period_log)],
                  check_val_every_n_epoch=period_log,
                  # Both same period in order to have the same logged values
                  precision=16,
                  # precision of the data. Check last ten seconds of:
                  # https://www.youtube.com/watch?v=d-2EHvJX03Y&ab_channel=PyTorchLightning
                  # does not affect the size of weights or the size used inside optimizer.
                  profile=True,  # or define profiler = AdvancedProfiler(); profiler = profiler; for more granularity.
                  callbacks=[early_stop_val_acc, early_stop_val_loss, early_stop_train_loss,
                             Progressbar_without_val_batch_update(refresh_rate=10),
                             chckpnt])
# num_sanity_val_steps Sanity check runs n validation batches before starting the training routine.
# auto_scale_batch_size="binsearch"
mnist_data_module = MyDataModule()

############################### Train the model
start_time = time.perf_counter()
trainer.fit(mnist_model, datamodule=mnist_data_module)
print("Total time training: ", time.perf_counter() - start_time, " seconds.")
#################


# To test a model, call trainer.test(model).
# Or, if you’ve just trained a model, you can just call trainer.test()
# and Lightning will automatically test using the best saved checkpoint (conditioned on val_loss).
# https://pytorch-lightning.readthedocs.io/en/latest/common/test_set.html
print(trainer.test(model=mnist_model, ckpt_path="best", dataloaders=mnist_data_module))

# for colab :
# # Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
