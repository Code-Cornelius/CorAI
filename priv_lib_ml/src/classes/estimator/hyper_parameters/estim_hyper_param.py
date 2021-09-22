import ast

import numpy as np
import pandas as pd
from priv_lib_estimator import Estimator

from priv_lib_ml.src.classes.architecture.fully_connected import Fully_connected_NN
from priv_lib_ml.src.classes.estimator.history.estim_history import Estim_history


class Estim_hyper_param(Estimator):
    # TODO 26/07/2021 nie_k: Refactor with factory pattern!!

    def __init__(self, df=None):
        super().__init__(df)

    @classmethod
    def from_folder(cls, path, metric_names, flg_time=False, compressed=True):
        """
        Semantics:
            Initialise an estim_hyper_param from a folder of estim_history.
        Args:
            path(str): The path to the folder.
            metric_names(list of str): The metric used for comparison.
            flg_time(bool): Flag to specify if the training time should be saved to the dataframe.
            compressed(bool): Flag to specify if compression is used.
        Returns:
            An Estim_hyper_param.
        """
        list_estimators = Estim_history.folder_json2list_estim(path, compress=compressed)
        return Estim_hyper_param.from_list(list_estimators, metric_names, flg_time)

    @classmethod
    def from_list(cls, estimators, metric_names, flg_time=False):
        """
        Semantics:
            Initialise an estim_hyper_param from a list of estim_history.
        Args:
            estimators(list of Estim_history): The estimators to be used.
            metric_names(list of str): The metrics used for comparison.
            flg_time(bool): Flag to specify if the training time should be saved to the dataframe.

        Returns:
            An Estim_hyper_param.
        """
        # collect the data from the estimators
        assert not isinstance(metric_names, str), "metric_names shall be a list of str."
        dataframe_information = \
            [Estim_hyper_param._get_dict_from_estimator(estimator, metric_names, flg_time) for estimator in estimators]

        # initialise the dataframe
        dataframe = pd.DataFrame(dataframe_information)
        return cls(dataframe)

    @staticmethod
    def _get_dict_from_estimator(estimator, metric_names, flg_time=False):
        estimator_dict = estimator.hyper_params.copy()
        for metric_name in metric_names:
            estimator_dict[metric_name] = estimator.get_best_value_for(metric_name)
        if flg_time:
            estimator_dict["train_time"] = estimator.get_time_best_fold()

        return estimator_dict

    def compute_number_params_for_fcnn(self):
        """
        Semantics:
            Computes the number of parameters for each entry and adds it to a new column.
        Requirements:
            Works for a fully connected NN, the estimator must contain the list of hidden sizes and the architecture.
            When the architecture for a row is not 'fcnn', the result will be NaN.
            If input size or output size are not present, they are assumed to be 1.
        Returns:
            Void.
        """
        assert 'architecture' in self.df.columns, "Cannot verify the architecture"
        assert 'list_hidden_sizes' in self.df.columns, "Cannot compute the number of parameters without" \
                                                       " list_hidden_sizes"

        if not 'input_size' in self.df.columns:
            print("No input size provided. Assume input size is 1.")
            self.df['input_size'] = [1] * self.df.shape[0]

        if not 'output_size' in self.df.columns:
            print("No output size provided. Assume output size is 1.")
            self.df['output_size'] = [1] * self.df.shape[0]

        # using apply to convert a list of strings into a list of lists. ast.literal_eval does the trick.
        # Reference: https://docs.python.org/3/library/ast.html
        self.df['list_hidden_sizes'] = self.df['list_hidden_sizes'].apply(ast.literal_eval)

        def compute_row(row):
            if row.architecture != 'fcnn':
                return np.nan
            return Fully_connected_NN.compute_nb_of_params(input_size=row.input_size,
                                                           list_hidden_sizes=row.list_hidden_sizes,
                                                           output_size=row.output_size)

        self.df['nb_of_params'] = self.df.apply(lambda row: compute_row(row), axis=1)
