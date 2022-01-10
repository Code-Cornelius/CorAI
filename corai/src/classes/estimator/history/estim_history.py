import json

import pandas as pd
import torch

from corai_error import Error_type_setter
from corai_estimator import Estimator
from corai_util.tools.src.decorator import decorator_delayed_keyboard_interrupt
from corai_util.tools.src.function_iterable import is_iterable


class Estim_history(Estimator):
    NAMES_COLUMNS = {'fold', 'epoch'}

    def __init__(self, df=None, metric_names=None, validation=True, hyper_params=None):
        if df is not None:
            super().__init__(df=df)
            return

        # metric names contain all ["L1","L4"...] but not the loss used for back prop.
        self.metric_names = metric_names
        self.validation = validation
        self.list_best_epoch = []  # list each entry corresponds to a fold
        self.list_train_times = []  # list of times it takes to train each fold
        self.hyper_params = hyper_params  # dict with serializable objects to be saved into a json.
        self.best_fold = -1  # negative strictly number means no best_fold found yet. Will be set in
        # train_kfold_a_fold_after_split

        df_column_names = self._generate_all_column_names()
        super().__init__(df=pd.DataFrame(columns=df_column_names))

    # section ######################################################################
    #  #############################################################################
    #  JSON constructor and saver.

    @decorator_delayed_keyboard_interrupt
    def to_json(self, path, compress=True, *kwargs):
        """
            Save an estimator to json as a compressed file.
        Args:
            compress: whether or not compression is applied
            path: The path where to store the estimator, with extension.

        Returns:`
            Void
        """
        # use delayed keyboard interrupt to assure the file saving is not interrupted
        attrs = {'metric_names': self.metric_names,
                 'validation': self.validation,
                 'best_epoch': self.list_best_epoch,
                 'time': self.list_train_times,
                 'hyper_params': self.hyper_params,
                 'best_fold': self.best_fold,
                 }
        ######## debugging advice:
        # if not serializable, check hyper_params that might contain wrong type objects.
        super().to_json(path, compress, attrs)

    @classmethod
    @decorator_delayed_keyboard_interrupt
    def from_json(cls, path, compressed=True):
        """
            Create estimator from previously stored json file
        Args:
            compressed: whether or not compression is applied
            path: The source path for the json, with extension.

        Returns:
            Void
        """

        attrs = super().from_json_attributes(path, compressed)
        estimator = super().from_json(path)

        estimator.metric_names = attrs['metric_names']
        estimator.validation = attrs['validation']
        estimator.list_best_epoch = attrs['best_epoch']
        estimator.hyper_params = attrs['hyper_params']
        estimator.best_fold = attrs['best_fold']
        estimator.list_train_times = attrs['time']

        estimator.to_json(path=path, compress=compressed)
        return estimator

    @classmethod
    @decorator_delayed_keyboard_interrupt
    def from_pl_logs(cls, log_path, checkpoint_path):
        """
        Semantics:
            Initialise an Estim_history from a python_light csv log file containing metrics and a checkpoint file.
        Args:
            log_path: Path to a csv file containing the metrics.
            checkpoint_path: Path to the checkpoint file of the model.

        Returns:
            An Estim_history.
        """
        estimator = super().from_csv(log_path)
        checkpoint = torch.load(checkpoint_path)

        # artificially insert fold column with 0 as value for compatibility
        estimator.df['fold'] = 0
        estimator.hyper_params = Estim_history.serialize_hyper_parameters(['hyper_parameters'])
        estimator.metric_names, estimator.validation = Estim_history.deconstruct_column_names(estimator.df.columns)

        # assume one fold case
        estimator.list_best_epoch = [checkpoint['epoch']]
        estimator.best_fold = 0
        estimator.list_train_times = []

        return estimator

    @staticmethod
    def deconstruct_column_names(column_names):
        """
        Semantics:
            Collect information about the metric names and whether validation is used from column names
        Assumption:
            - The column name has ["train", "training", "val", "validation"] separated by "_" either
            before or after the metric name. For example, train_loss is valid, but trainLoss is not.
            - If a metric has both training and validation values the metric name used will be the same
        Args:
            column_names([str]): List of column names.
        Returns:
            metric_names([str]): List of the metric names.
            validation(bool): Flag to specify validation.
        """
        val_keywords = ["val", "validation"]
        train_keywords = ["train", "training"]
        ignore = ["epoch", "fold", "step"]
        validation = False
        metric_names = set()
        for name in column_names:
            if name in ignore: # we continue to not add these columns to the metric_names.
                continue
            components = name.split("_")
            if any(val_keyword in components for val_keyword in val_keywords):
                validation = True

            # remove keywords from metric name
            metric_name = '_'.join(
                [word for word in components  # components is the name split.
                 if word not in val_keywords + train_keywords]) # check if word is in the keywords. If it is, discard.

            metric_names.add(metric_name) # there might be train_loss, val_loss. We only want it once.

        return list(metric_names), validation

    @staticmethod
    def serialize_hyper_parameters(hyper_parameters):
        if isinstance(hyper_parameters,dict):
            for key,value in hyper_parameters.items():
                if is_iterable(value):
                    hyper_parameters[key] = [Estim_history.serialize_hyper_parameters([hp_value]) for hp_value in value]
                    # : calls with [.] to make it iterable.
                elif not Estim_history.is_jsonable(value):
                    components = str(value).split(' ')
                    hyper_parameters[key] = components[2] # 2nd elmnt  which is the one we want
            return hyper_parameters

        else: # iterable case
            for i, elmnt in enumerate(hyper_parameters):
                if is_iterable(elmnt):
                    hyper_parameters[i] = [Estim_history.serialize_hyper_parameters([smaller_elmt]) for smaller_elmt in elmnt]
                elif not Estim_history.is_jsonable(elmnt):
                    components = str(elmnt).split(' ')
                    hyper_parameters[i] = components[2]  # 2nd elmnt  which is the one we want
                if len(hyper_parameters) == 1: # we unpack the list if it was not a list in the first place
                    hyper_parameters = hyper_parameters[0]
            return hyper_parameters


    @staticmethod
    def is_jsonable(x):
        # todo move to somewhere else in utils.
        # reference
        # https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python

        try:
            json.dumps(x)
            return True
        except:
            return False



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
            fold_time (float): the time it took to train one fold, seconds.
            period_kept_data(int): Keep only elements on indices that are multiple of period_kept_data.
                If None dataframe is not affected. best_epoch will not be removed.
        Returns:
            Void
        """
        self.list_best_epoch.append(fold_best_epoch)  # append to the best_epochs, the current folds' best epoch.
        self.list_train_times.append(fold_time)  # append the time to run current epoch
        history = pd.DataFrame(history)

        # Remove every split index from dataframe before appending
        if period_kept_data:
            # save best row
            row = history.loc[fold_best_epoch]
            n = history.loc[:, 'epoch'].max() + 1

            index_to_remove = [j for i in range(0, n, period_kept_data) for j in
                               range(i + 1, min(i + period_kept_data, n))]

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
