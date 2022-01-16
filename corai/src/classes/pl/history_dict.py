import collections

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
        - adaptor from a dictionnary with results to estim_history class.
    """

    # for this class to work properly, when one logs from the validation_step,
    # it should contain in the string (name) one sub-string from the list below.
    val_keywords = ['val', 'validation']

    def __init__(self, aplot_flag=False, frequency_epoch_logging=1, metrics_plot=['train_loss', 'val_loss']):
        # frequency_epoch_logging = check_val_every_n_epoch from trainer.
        # metrics_plot are the metrics that will be plot at each iteration of the evolution of the loss. Iterable.
        super().__init__()

        self.hyper_params = None
        self.history = collections.defaultdict(list)
        # The defaultdict will create an entry with an empty list if they key is missing when trying to access
        self.freq_epch = frequency_epoch_logging
        if aplot_flag:
            self.aplot = corai_plot.APlot(how=(1, 1))
            self.colors = corai_plot.AColorsetDiscrete("Dark2")
        else:
            self.aplot = None

        self.metrics_name_for_plot = metrics_plot
        # adding a zero to the metrics with validation inside
        # (this is because pytorch lightning does validation after the optimisation step).
        for name in self.metrics_name_for_plot:  # this is done such that we can use fetch_score method in the plotting method.
            if any(val_keyword in name for val_keyword in History_dict.val_keywords):
                self.history[name] = [0]
            else:
                self.history[name] = []

    @property
    def name(self):
        return "Corai_History_Dict_Logger"

    @property
    def version(self):
        return "1.0"

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
                        not self.history['epoch'][-1] == metric_value):
                    # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
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
            raise KeyError("The key does not exist in history.")

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
        losses = self.fetch_score(self.metrics_name_for_plot)
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
                                                         'label': self.metrics_name_for_plot[i]},
                                        dict_ax={'title': "Dynamical Image of History Training", 'xlabel': 'Epochs',
                                                 'ylabel': 'Loss',
                                                 'yscale': 'log'})
            self.aplot.show_legend()
            self.aplot.show_and_continue()

    def to_estim_history(self, checkpoint, train_time):
        """
            Transform a history dict to an Estim_history using a checkpoint.
        Args:
            checkpoint: Pytorch lightning checkpoint.
            train_time: The time taken for the training.
        Returns:
            An Estim_history.

        Preconditions:
            The column name has ["train", "training", "val", "validation"] separated by "_" either
                    before or after the metric name. For example, train_loss is valid, but trainLoss is not.
        """

        # from init, there is one entry more in the training metrics.
        # We delete the last entry of each list with validation in the name of the metric.
        for key in self.history:
            if any(val_keyword in key for val_keyword in History_dict.val_keywords):  # case val
                self.history[key] = self.history[key][:-1]  # remove last entry of the list
        df = pd.DataFrame(self.history)
        estimator = Estim_history(df=df)

        estimator.df['fold'] = 0  # estimator have been written so the fold column exist.

        checkpoint = torch.load(checkpoint.best_model_path)
        estimator.hyper_params = Estim_history.serialize_hyper_parameters(self.hyper_params)
        estimator.metric_names, estimator.validation, estimator.df.columns = \
            Estim_history.deconstruct_column_names(estimator.df.columns)

        # assumes one fold case
        estimator.list_best_epoch = [checkpoint['epoch']]
        estimator.best_fold = 0
        estimator.list_train_times = [train_time]

        return estimator
