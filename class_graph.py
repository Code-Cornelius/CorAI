# normal libraries
from abc import abstractmethod

import pandas as pd

# my libraries
from class_estimator import Estimator
from plot_functions import APlot

class Graph:

    def __init__(self, estimator, separators=None):
        self.estimator = estimator
        self.separators = separators

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        estimator = Estimator(pd.read_csv(path))
        # get the max value which is M-1
        return cls(estimator)

    def generate_title(self, names, values):
        title = "Histogram"
        for (name, value) in zip(names, values):
            title += ", " + name + " = " + str(value)
        return title

    @abstractmethod
    def get_range(self, key, mean):
        pass

    @abstractmethod
    def get_param_info(self, key, mean):
        pass

    @abstractmethod
    def get_fig_dict(self, separators, key):
        pass

    def histogram_of_realisations_of_estimator(self, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.slice_DF(separators)

        for key in keys:
            data = global_dict.get_group(key)['value']
            mean = data.mean()
            data = data.values
            plot = APlot()
            param_dict = self.get_param_info(key, mean)
            fig_dict = self.get_fig_dict(separators, key)
            plot.hist(data=data, param_dict_hist=param_dict, fig_dict=fig_dict)


    def estimation_hawkes_parameter_over_time(self, **kwargs):
        pass

