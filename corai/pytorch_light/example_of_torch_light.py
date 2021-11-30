# example from: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# adds on from https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
import time

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from corai.pytorch_light.classes_mnist import *

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 256

seed_everything(42, workers=True)

# Init our model
mnist_model = MNISTModel(learning_rate=0.001)

# Init the Early Stopper https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.early_stopping.html#pytorch_lightning.callbacks.early_stopping.EarlyStopping
early_stop_val_acc = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=10, verbose=False, mode="max", )
# stopping_threshold = 0.9)
early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10, verbose=False, mode="min", )
early_stop_train_loss = EarlyStopping(monitor="train_loss", min_delta=-0.001, patience=10, verbose=False, mode="min", )

# Initialize a trainer # logger : https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-frequency
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=100,
                  # progress_bar_refresh_rate=50, # Ignored when a custom progress bar is passed to callbacks.
                  # progress bar over the batches, but is deprecated needs to find alternative.
                  log_every_n_steps=1, precision=16,
                  callbacks=[early_stop_val_acc, early_stop_val_loss, early_stop_train_loss,
                             LitProgressBar(refresh_rate=10),
                             ModelCheckpoint(monitor = "val_acc", mode = "max", verbose = True)])
mnist_data_module = MyDataModule()

#################
# Train the model
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
