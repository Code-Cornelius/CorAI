# from metric.Metric import Metric
from priv_lib_util.tools import function_iterable

from src.nn_classes.metric.metric import Metric


class NNTrainParameters:

    def __init__(self, batch_size, epochs, device, criterion, optim_wrapper, metrics=()):
        """

        Args:
            batch_size: batch size can be too big, will not have an impact in kfold_training.
            epochs:
            device:
            criterion: should be not normalised. All metrics and losses are rescaled inside by batch size.
            optimiser_wrapper: contains the optimiser object with the parameters to initialise it.
                It can also contain a scheduler for updating the learning rate
            metrics:  iterable containing objects of type Metric.
            The history is computed by computing over each batch and at the end dividing by total length of data.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.criterion = criterion
        self.optim_wrapper = optim_wrapper

        # iterable containing objects of type Metric
        self.metrics = metrics

    # SETTERS GETTERS
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        if isinstance(new_batch_size, int) and new_batch_size >= 0:
            self._batch_size = new_batch_size
        else:
            raise TypeError(f"Argument is not an unsigned int.")

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, new_epochs):
        if isinstance(new_epochs, int) and new_epochs >= 0:
            self._epochs = new_epochs
        else:
            raise TypeError(f"Argument is not an unsigned int.")

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, new_criterion):
        self._criterion = new_criterion

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, new_optimiser):
        self._optimiser = new_optimiser

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics):
        assert function_iterable.is_iterable(new_metrics), "argument should be iterable."
        for new_metric in new_metrics:
            if isinstance(new_metric, Metric):
                pass
            else:
                raise TypeError(f"Argument is not a metric.")
        self._metrics = new_metrics
