import collections

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class History_dict(LightningLoggerBase):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list)
        # The defaultdict will create an entry with an empty list if they key is missing when trying to access

    @property
    def name(self):
        return "Logger_custom_plot"

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
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)
        return

    def log_hyperparams(self, params):
        pass


######### the plotting would for now look like:

"""
loss = loggers[1].history['train_loss']
loss_epochs = loggers[1].history['epoch']
aplot = corai_plot.APlot(how=(1, 1))
aplot.uni_plot(0, loss_epochs, loss,
               dict_plot_param={'color': None, 'linestyle': '-', 'linewidth': 2.5, 'markersize': 0.,
                                'label': 'Training Loss'},
               dict_ax={'title': "History Training", 'xlabel': 'Epochs', 'ylabel': 'Loss', 'yscale': 'log'})
aplot.show_legend()"""