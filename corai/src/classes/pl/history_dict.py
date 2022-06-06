import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

import corai_plot
from corai.src.classes.estimator.history import Estim_history


class History_dict(LightningLoggerBase):
    """
    Useful class that is at the same time:
        - helper to save the history of training somewhere on the heap instead of in a file.
        - helper to plot the evolution of the losses DURING training,
        - adaptor from a dictionary with results to estim_history class,
        - stores the hyperparameters of the model.

    There is no particular rule to use it.
    However, we noticed that validation metrics are shifted compared to training.
    We provide a solution, whenever the name of the validation metrics contain the words 'val' or 'validation'.
    It is encourage to do include these words in the naming of the metric.
    We also recommend writing the names of the metrics as: name_type where type can be training or validation.
    This is useful when one converts the history dict into an estimator history.
    """

    # for this class to work properly, when one logs from the validation_step,
    # it should contain in the string (name) one sub-string from the list below.
    val_keywords = ['val', 'validation']

    def __init__(self, metrics, aplot_flag=False, frequency_epoch_logging=1):
        # frequency_epoch_logging = check_val_every_n_epoch from trainer.
        # metrics are the metrics that will be plot at each iteration of the evolution of the loss. Iterable.
        # It also contains the metrics that are stored. If the metrics name do not agree with what is logged,
        # there will be a bug.
        super().__init__()

        self.hyper_params = {}  # by default a dictionary because it is the way it is stored.
        self.history = {}
        # The defaultdict will create an entry with an empty list if they key is missing when trying to access
        self.freq_epch = frequency_epoch_logging
        if aplot_flag:
            self.aplot = corai_plot.APlot(how=(1, 1))
            self.colors = corai_plot.AColorsetDiscrete("Dark2")
        else:
            self.aplot = None

        self.metrics = metrics
        # adding a nan to the metrics with validation inside
        # (this is because pytorch lightning does validation after the optimisation step).
        # Hence, there is always a shift between the two values.
        # also we initialize the history.
        self.history["epoch"] = []
        for name in self.metrics:  # this is done such that we can use fetch_score method in the plotting method.
            if any(val_keyword in name for val_keyword in History_dict.val_keywords):
                self.history[name] = [np.NAN]  # add a nan to the validation metrics.
            else:
                self.history[name] = []

    @property
    def name(self):
        return "Corai_History_Dict_Logger"

    @property
    def version(self):
        return "V1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        Args:
            metrics (dictionary): contains at least the metric name (one metric), the value and the epoch nb.
            step: number of steps went through (validation or training).

        """

        # frequency check. This ensures that the number of logs for train and validation are equal.
        if ((metrics['epoch'] + 1) % self.freq_epch) != 0:
            # + 1 accounts for the shift with the frequency of validation step, in trainer
            # ~~~~~~~~~~~~~~~~~~~~~~ check_val_every_n_epoch = self.freq_epch ~~~~~~~~~~~~~~~~~~
            return

        # fetch all metrics. We use append (complexity amortized O(1)).
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else:  # case epoch. We want to avoid adding multiple times the same.
                # It happens when multiple losses are logged.
                if (not len(self.history['epoch']) or  # len == 0:
                        not self.history['epoch'][-1] == (metric_value + 1)):
                    # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value + 1)  # shift
                else:
                    pass
        self.plot_history_prediction()
        return

    def log_hyperparams(self, params, *args, **kwargs):
        self.hyper_params = params

    def _get_history_one_key(self, key):
        # Removes last point from validation scores.
        if key in self.history:
            if any(val_keyword in key for val_keyword in History_dict.val_keywords):
                return self.history[key][:-1]  # returns and stop iter. All apart from last point.

            return self.history[key]  # did not the val key word.

        else:
            raise KeyError(f"The key {key} does not exist in history. "
                           f"If key is supposed to exist, has it been passed to the constructor of the logger?")

    def fetch_score(self, keys):
        """
        Semantics:
            Gets the score if exists in the history. Removes last point from validation scores.

        Args:
            keys (str or list<str>): the keys to fetch the result.

        Returns:
            list of lists of score.

        """
        # string or list of strings
        if isinstance(keys, str):
            return [self._get_history_one_key(keys)]
        else:
            res = [0] * len(keys)
            for i, key in enumerate(keys):
                res[i] = self._get_history_one_key(key)
            return res

    def plot_history_prediction(self):
        epochs_loss, = self.fetch_score(['epoch'])
        losses = self.fetch_score(self.metrics)
        len_loss = [len(lst) for lst in losses]
        if self.aplot is not None and max(len_loss) == min(len_loss) == len(epochs_loss):
            # plot the prediction:
            self.aplot._axs[0].clear()

            # plot losses
            if len(epochs_loss) > 1:  # make the test so it does not plot in the case of empty loss.
                for i, (color, loss) in enumerate(zip(self.colors, losses)):
                    self.aplot.uni_plot(0, epochs_loss, loss,
                                        dict_plot_param={'color': color, 'linestyle': '-', 'linewidth': 2.5,
                                                         'markersize': 0.,
                                                         'label': self.metrics[i]},
                                        dict_ax={'title': "Dynamical Image of History Training", 'xlabel': 'Epochs',
                                                 'ylabel': 'Loss',
                                                 'yscale': 'log'})
            self.aplot.show_legend()
            self.aplot.show_and_continue()

    def to_estim_history(self, checkpoint, train_time):
        """
            Transform a history dict to an Estim_history using a checkpoint.
            The history_dict could have multiple different column names, like train_loss and val_loss.
            This naming is not suited for estim_history, and so we convert the namings with the function
            `deconstruct_column_names`
        Args:
            checkpoint: Pytorch lightning checkpoint. It has the model inside.
            train_time: The time taken for the training.
        Returns:
            An Estim_history.
        """
        checkpoint = torch.load(checkpoint.best_model_path)

        # from init, there is one entry more in the validation metrics.
        # We delete the last entry of each list with validation in the name of the metric.
        list_length_assertion = []  # fetch the different length to assert it
        for key in self.history:
            if any(val_keyword in key for val_keyword in History_dict.val_keywords):  # case val
                self.history[key] = self.history[key][:-1]  # remove last entry of the list

            list_length_assertion.append(len(self.history[key]))

        assert max(list_length_assertion) == min(list_length_assertion), "Some Metrics have more values than others."

        df = pd.DataFrame(self.history)
        df['fold'] = 0  # estimators require a fold column.

        hyper_params = Estim_history.serialize_hyper_parameters(self.hyper_params)  # put them in correct form
        # we rename the columns so they suit the standard of estimators.
        metric_names, columns, validation = Estim_history.deconstruct_column_names(df.columns)
        df.columns = columns
        estimator = Estim_history(df=df, metric_names=metric_names, validation=validation, hyper_params=hyper_params)

        # assumes one fold case
        estimator.list_best_epoch.append(checkpoint['epoch'])
        estimator.list_train_times.append(train_time)

        estimator.best_fold = 0

        return estimator
