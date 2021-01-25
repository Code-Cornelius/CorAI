# normal libraries
from abc import abstractmethod

from priv_lib_util.tools import function_str
import numpy as np  # maths library and arrays
from priv_lib_estimator.src.estimator.estimator import Estimator
from priv_lib_estimator.src.plot_estimator.plot_estimator import Plot_estimator

# my libraries
from priv_lib_plot import APlot


# errors:


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Histogram_estimator(Plot_estimator):
    # abstract nb_of_bins parameter
    @property
    @abstractmethod
    def NB_OF_BINS(self):
        pass

    def __init__(self, estimator, separators=None, *args, **kwargs):
        super().__init__(estimator, separators, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data

    # section ######################################################################
    #  #############################################################################
    # plot

    @staticmethod
    def get_range(key, mean):
        return None

    @abstractmethod
    def get_dict_param(self, key, mean):
        pass

    @abstractmethod
    def get_dict_fig(self, separators, key):
        pass

    def draw(self, separators=None):
        separators, global_dict, keys = super().draw(separators=separators)

        for key in keys:
            data = global_dict.get_group(key)['value']
            mean = data.mean()
            data = data.values
            plot = APlot()
            param_dict = self.get_dict_param(key, mean)
            fig_dict = self.get_dict_fig(separators, key)
            plot.hist(data=data, dict_param_hist=param_dict, dict_ax=fig_dict)
            name_file = ''.join([priv_lib_util.tools.function_str.tuple_to_str(key, ''), 'histogram'])
            plot.save_plot(name_save_file=name_file)
