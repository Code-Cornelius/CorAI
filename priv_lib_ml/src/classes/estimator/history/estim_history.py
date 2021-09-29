import numpy as np
import pandas as pd
import os

from priv_lib_error import Error_type_setter
from priv_lib_estimator import Estimator

from priv_lib_ml.src.plot.nn_plots import nn_errors_compute_mean

class Estim_history(Estimator):
    NAMES_COLUMNS = {'fold', 'epoch'}

    def __init__(self, df=None, metric_names=None, validation=True, hyper_params=None):
        if df is not None:
            super().__init__(df=df)
            return

        # metric names contain all ["L1","L4"...] but not the loss used for back prop.
        self.metric_names = metric_names
        self.validation = validation
        self.list_best_epoch = [] # list each entry corresponds to a fold
        self.list_train_times = [] # list of times it takes to train each fold
        self.hyper_params = hyper_params
        self.best_fold = -1  # negative strictly number means no best_fold found yet. Will be set in
        # train_kfold_a_fold_after_split

        self.err_computed = False  # flag that indicates whether all losses are stored.

        df_column_names = self._generate_all_column_names()
        super().__init__(df=pd.DataFrame(columns=df_column_names))

    # section ######################################################################
    #  #############################################################################
    #  JSON constructor and saver.

    def to_json(self, path, compress=True, *kwargs):
        """
            Save an estimator to json as a compressed file.
        Args:
            compress: whether or not compression is applied
            path: The path where to store the estimator, with extension.

        Returns:`
            Void
        """
        attrs = {
            'validation': self.validation,
            'best_epoch': self.list_best_epoch,
            'hyper_params': self.hyper_params,
            'best_fold': self.best_fold,
            'err_computed': self.err_computed,
            'time': self.list_train_times
        }

        if self.err_computed:
            add = {'train_mean_loss_L1': self.train_mean_loss_L1,
                   'train_mean_loss_L2': self.train_mean_loss_L2,
                   'train_mean_loss_Linf': self.train_mean_loss_Linf,
                   'test_mean_loss_L1': self.test_mean_loss_L1,
                   'test_mean_loss_L2': self.test_mean_loss_L2,
                   'test_mean_loss_Linf': self.test_mean_loss_Linf}
            attrs.update(add)

        super().to_json(path, compress, attrs)

    @classmethod
    def from_json(cls, path, compressed=True):
        """
            Create estimator from previously stored json file
        Args:
            compressed: whether or not compression is applied
            path: The source path for the json

        Returns:
            Void
        """
        attrs = super().from_json_attributes(path, compressed)
        estimator = super().from_json(path)

        estimator.validation = attrs['validation']
        estimator.list_best_epoch = attrs['best_epoch']
        estimator.hyper_params = attrs['hyper_params']
        estimator.best_fold = attrs['best_fold']
        estimator.err_computed = attrs['err_computed']
        estimator.list_train_times = attrs['time']

        if estimator.err_computed:  # flag that indicates whether all losses are stored.
            estimator.train_mean_loss_L1 = attrs['train_mean_loss_L1']
            estimator.train_mean_loss_L2 = attrs['train_mean_loss_L2']
            estimator.train_mean_loss_Linf = attrs['train_mean_loss_Linf']
            estimator.test_mean_loss_L1 = attrs['test_mean_loss_L1']
            estimator.test_mean_loss_L2 = attrs['test_mean_loss_L2']
            estimator.test_mean_loss_Linf = attrs['test_mean_loss_Linf']

        estimator.to_json(path=path, compress=compressed)
        return estimator

    def get_col_metric_names(self):
        """
        Semantics:
            Generate the names for the columns which hold information about the metrics.

        Returns:
            A list of strings representing the column names based on metric names and validation flag

        Dependency:
            self.metric_names
            self.validation
            The order of the list is important, and is used in relplot history
            (first the training losses, then validation losses).
        """
        df_column_names = [Estim_history.generate_column_name("loss")]

        for metric_name in self.metric_names:
            df_column_names.append(Estim_history.generate_column_name(metric_name))

        if self.validation:  # validation columns
            df_column_names.append(Estim_history.generate_column_name("loss", validation=True))

            for metric_name in self.metric_names:
                df_column_names.append(Estim_history.generate_column_name(metric_name, validation=True))

        return df_column_names

    @staticmethod
    def generate_column_name(base_name, validation=False):
        """
        Semantics:
            Generate the column name based on a metric name and its use case (training or validation)

        Args:
            base_name: the name of the metric
            validation: boolean representing whether the metric is a result of validation or training

        Returns:
            The formatted column name
        """
        return f"{base_name}_" + ('validation' if validation else 'training')

    def _generate_all_column_names(self):
        """
        Generate all the column names for the dataframe
        Returns:
            A list of all the column names for the dataframe (Including the base columns)

        Dependency:
            self.get_col_metric_names
            self.metric_names
            self.validation
        """
        df_column_names = list(Estim_history.NAMES_COLUMNS.copy()) + self.get_col_metric_names()

        return df_column_names

    def append(self, history, fold_best_epoch, fold_time, period_kept_data=None, *args, **kwargs):
        """
            Append information from history to the estimator
        Args:
            history (dict): history of the training, a certain shape that is agreed upon in the training,
            fold_best_epoch (int): best epoch for a model,
            fold_time (int): the time it took to train one fold, seconds.
            period_kept_data(int): Keep only elements on indices that are multiple of period_kept_data.
                If None dataframe is not affected. best_epoch will not be removed.
        Returns:
            Void
        """
        self.list_best_epoch.append(fold_best_epoch) # append to the best_epochs, the current folds' best epoch.
        self.list_train_times.append(fold_time) # append the time to run current epoch
        history = pd.DataFrame(history)

        # Remove every split index from dataframe before appending
        if period_kept_data:
            # save best row
            row = history.loc[fold_best_epoch]
            n = history.loc[:, 'epoch'].max() + 1

            index_to_remove = [j for i in range(0, n, period_kept_data) for j in range(i+1, min(i+period_kept_data, n))]

            index_to_remove.append(fold_best_epoch)

            history = history.drop(index_to_remove)
            history = history.append(row)

        super().append(history, *args, **kwargs)
        return


    def _index_mask(self, fold, epoch):
        return (self.df.loc[:, 'fold'] == fold) & (self.df.loc[:, 'epoch'] == epoch)

    def _fold_mask(self, fold):
        return self.df.loc[:, 'fold'] == fold

    @property
    def nb_folds(self):
        return self.df.loc[:, 'fold'].max() + 1

    @property
    def nb_epochs(self):
        return self.df.loc[:, 'epoch'].max() + 1

    def slice_best_fold(self):
        """
        Semantics:
            Slice the dataframe to only store the best fold.

        Returns:
            Void
        """
        # wip for now we de not slice the two lists, as it is more complicated to keep track later on that we sliced,
        #  and that the best_fold for example do not correspond to the index of the two lists.
        # self.list_train_times = [self.list_best_epoch[self.best_fold]]
        # self.list_best_epoch = [self.list_best_epoch[self.best_fold]]
        self.df = self.df.loc[self._fold_mask(self.best_fold)]

    # section ######################################################################
    #  #############################################################################
    # SETTERS GETTERS

    def get_values_fold_epoch_col(self, fold, epoch, column):
        index = self._index_mask(fold, epoch)
        return self.df.loc[:, column][index].values[0]

    def get_values_fold_col(self, fold, column):
        index = self._fold_mask(fold)
        return self.df.loc[:, column][index].values

    def get_values_col(self, column):
        return self.df.loc[:, column].values

    def get_best_value_for(self, column):
        try:
            epoch = self.list_best_epoch[self.best_fold]
            # best_fold and best_epoch is correctly set as soon as one calls train_kfold_a_fold_after_split.
            return self.get_values_fold_epoch_col(self.best_fold, epoch, column)
        except KeyError:
            raise KeyError(f"Column {column} does not belong to the columns' names of the Estimator.")

    def get_time_best_fold(self):
        return self.list_train_times[self.best_fold]

    @property
    def best_fold(self):
        return self._best_fold

    @best_fold.setter
    def best_fold(self, new_best_fold):
        if isinstance(new_best_fold, int):
            self._best_fold = new_best_fold
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    def err_compute_best_net(self, net, train_X, train_Y, testing_X=None, testing_Y=None, device='cpu'):
        self.err_computed = True  # flag that indicates whether all losses are stored.

        (trainL1, trainL2, trainLinf, testL1,
         testL2, testLinf) = nn_errors_compute_mean(net=net, device=device,
                                                    train_X=train_X, train_Y=train_Y,
                                                    testing_X=testing_X, testing_Y=testing_Y)

        self.train_mean_loss_L1 = trainL1
        self.train_mean_loss_L2 = trainL2
        self.train_mean_loss_Linf = trainLinf
        self.test_mean_loss_L1 = testL1
        self.test_mean_loss_L2 = testL2
        self.test_mean_loss_Linf = testLinf
        return (trainL1, trainL2, trainLinf, testL1,
                testL2, testLinf)

    def print_err(self):
        print("Relative Mean Training L1 Error: {:e}%.".format(self.train_mean_loss_L1 * 100))
        print("Relative Mean Training L2 Error: {:e}%.".format(self.train_mean_loss_L2 * 100))
        print("Relative Mean Training Linf Error: {:e}%.".format(self.train_mean_loss_Linf * 100))
        if self.test_mean_loss_L1:  # != 0
            print("Relative Mean Testing L1 Error: {:e}%.".format(self.test_mean_loss_L1 * 100))
            print("Relative Mean Testing L2 Error: {:e}%.".format(self.test_mean_loss_L2 * 100))
            print("Relative Mean Testing Linf Error: {:e}%.".format(self.test_mean_loss_Linf * 100))
        return

