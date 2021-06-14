# normal libraries
from abc import abstractmethod

import priv_lib_util
from priv_lib_util.tools import function_str
import numpy as np  # maths library and arrays
from priv_lib_estimator.src.estimator.estimator import Estimator
from priv_lib_estimator.src.plot_estimator.plot_estimator import Plot_estimator
from priv_lib_util.tools.src.function_dict import filter
# my libraries
from priv_lib_plot import APlot


# errors:


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Histogram_estimator(Plot_estimator):
    """
    Abstract class

    Redefine:
        get_dict_fig
        get_dict_plot_param
    """
    @property
    @abstractmethod
    def NB_OF_BINS(self):
        # abstract nb_of_bins parameter
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
    def get_range(key, mean, std):
        # method to overload if one wants to fix a certain range for the histogram.
        """

        Args:
            key:
            mean:
            std:

        Returns:

        Examples:
            return (mean - 2 * std, mean + 2 * std)


        """
        return None

    @abstractmethod
    def get_dict_plot_param(self, key, mean, std):
        """
        Dictionary with the parameters for the plot
        Args:
            key:
            mean:
            std:


        Returns:

        Examples:
                dict_param = {'bins': self.NB_OF_BINS,
                  'label': 'Histogram',
                  'color': 'green',
                  'range': self.get_range(key, mean, std),
                  'cumulative': True}
                return dict_param
        """
        pass

    @abstractmethod
    def get_dict_fig(self, separators, key):
        """

        Args:
            separators:
            key:

        Returns:

        Examples:
            title = self.generate_title(parameters=separators, parameters_value=key,
                                        before_text="Histogram")
            fig_dict = {'title': title,
                        'xlabel': "x",
                        'ylabel': "y"}
            return fig_dict

        """
        pass

    def draw(self, feature_to_draw, separators=None, separator_filter=None, save_plot=True):
        """

        Args:
            feature_to_draw:
            separators:
            separator_filter:
            save_plot:

        Returns:

        """
        separators, global_dict, keys = super().draw(separators_plot=separators)
        keys = filter(separators, keys, separator_filter)
        for key in keys:
            data = global_dict.get_group(key)[feature_to_draw]
            mean = data.mean()
            std = data.std()
            data = data.values
            plot = APlot()
            param_dict = self.get_dict_plot_param(key, mean, std)
            fig_dict = self.get_dict_fig(separators, key)
            plot.hist(data=data, dict_param_hist=param_dict, dict_ax=fig_dict)
            name_file = ''.join([priv_lib_util.tools.function_str.tuple_to_str(key, ''), 'histogram'])

            if save_plot:
                plot.save_plot(name_save_file=name_file)
