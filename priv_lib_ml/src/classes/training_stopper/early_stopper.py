from abc import abstractmethod, ABCMeta

import numpy as np
from priv_lib_error import Error_type_setter

DEBUG = False


class Early_stopper(metaclass=ABCMeta):
    """
    Abstract class of an early stopper. Given to a training, allows for stopping earlier with respect to some criteria.
    Used to stop training earlier given some criteria. Undefined behaviour when patience is greater than nb of epoch.

    The requirements for children class is:
        redefine _is_early_stop

    In order to use it, one initializes it and then call it with the corresponding data.
    The call returns whether or not we should "early stop" training.

    Fields of an early_stopper:
        _patience
        _silent
        _delta
        _print_func
        _counter
        _lowest_loss
        _early_stopped = False
        has_improved_last_epoch = True

    One should check two things. Early_stopped allows to know whether we early_stopped.
    has_improved_last_epoch is a flag showing whether the stopper wishes to save the model or not (because the new model is better since the last check).

    Pre-condition:
        history in training must exists. Has the form:

            history = {
                'training': {},
                'validation': {}
            }
            history['training']['loss'] = np.zeros((nb_split, parameters_training.epochs))
            history['validation']['loss'] = np.zeros((nb_split, parameters_training.epochs))

            for metric in parameters_training.metrics:
                history['training'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
                history['validation'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
    """

    def __init__(self, tipee, metric_name, patience=50, silent=True, delta=0.1):
        """
        Args:
            tipee (str): the type of early stopper, used for accessing history in NN training.
            'training', 'validation' are supported.
            metric_name (str): the named used to identify the results of the metric in history
            patience (int): How long the stopper waits for improvement of the criterion.
            silent (bool): verbose
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. In percent.
                            Default: 0.1
        """
        self._patience = patience
        self._silent = silent
        self._counter = 0 # private counter that is compared to patience
        self._lowest_loss = np.Inf
        self._delta = delta

        # status of the early stopper
        self._early_stopped = False
        self.has_improved_last_epoch = True

        # for retrieving information from the history, private arguments
        self._tipee = tipee
        self._metric_name = metric_name

    def __call__(self, net, history, epoch):
        """
        Semantics:
            Check if the condition for early stopping are satisfied.
            Condition is _is_early_stop. The history should be given as a dict of dict (validation/training; metric type).
            If _is_early_stop == true, then has_improved_last_epoch = False.
            If _is_early_stop == false, then has_improved_last_epoch = True.
        Args:
            net:
            history:
            epoch:

        Returns:
            self._counter >= self._patience:

        """
        # method implemented child specific class, case should stop.
        if self._is_early_stop(history[self._tipee][self._metric_name], epoch):
            self._counter += 1
            self.has_improved_last_epoch = False  # : flag giving information about the performance of the NN
            if DEBUG:
                print(f'EarlyStopping counter: {self._counter} out of {self._patience}')

            # early stop triggered
            if self._counter >= self._patience:
                self._early_stopped = True
                return True
        # should not stop / has improved.
        else:
            self.has_improved_last_epoch = True  # : flag giving information about the performance of the NN
            self._lowest_loss = min(history[self._tipee][self._metric_name][epoch], self._lowest_loss)
            self._counter = 0
        return False

    @abstractmethod
    def _is_early_stop(self, losses, epoch):
        """
        Test for earling stopping regarding some criteria.
        Const method

        Args:
            losses: data for criteria, list, needs to be iterable.
            epoch: current epoch.

        Returns:
             boolean answering the question should we stop early at the current epoch.

        """
        pass

    def is_stopped(self):
        return self._early_stopped

    def is_validation(self):
        return self._tipee == 'validation'

    def reset(self):
        """ Allows to reset the log of the early stopper, for multiple usage.
         For example, in kfold."""
        self._early_stopped = False  # if we train again, then we reset early_stopped.
        self.has_improved_last_epoch = True
        self._lowest_loss = np.Inf
        self._counter = 0

    @property
    def _tipee(self):
        return self.__tipee  # un-mangling

    @_tipee.setter
    def _tipee(self, new__tipee):
        if isinstance(new__tipee, str):
            self.__tipee = new__tipee
        else:
            raise Error_type_setter(f'Argument is not an {str(str)}.')

    @property
    def _metric_name(self):  # un-mangling
        return self.__metric_name

    @_metric_name.setter
    def _metric_name(self, new_metric_name):
        if isinstance(new_metric_name, str):
            self.__metric_name = new_metric_name
        else:
            raise Error_type_setter(f'Argument is not an {str(str)}.')
