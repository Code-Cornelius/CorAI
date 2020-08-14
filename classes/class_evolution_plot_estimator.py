# normal libraries
from abc import abstractmethod
import numpy as np  #maths library and arrays
import statistics as stat
import pandas as pd  #dataframes
import seaborn as sns  #envrionement for plots
from matplotlib import pyplot as plt  #ploting 
import scipy.stats  #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time  #allows to time event
import warnings
import math  #quick math functions
import cmath  #complex functions

# my libraries
import classical_functions
import decorators_functions
import financial_functions
import functions_networkx
from plot_functions import APlot
import recurrent_functions
from classes.class_estimator import Estimator
from classes.class_graph_estimator import Graph_Estimator

# errors:
import errors.Error_convergence
import errors.Warning_deprecated
import errors.Error_forbidden

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Evolution_plot_estimator(Graph_Estimator):
    def __init__(self, estimator, separators=None):
        super().__init__(estimator = estimator, separators = separators)

    @abstractmethod
    def get_evolution_true_value(self, data):
        pass

    @abstractmethod
    def get_evolution_plot_data(self, data):
        pass

    @abstractmethod
    def get_evolution_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute,
        and we focus on data given by column str, computing the means and returning an array.

        :param data:
        :param str:
        :return:
        '''
        pass

    @abstractmethod
    def get_dict_fig_evolution_parameter_over_time(self, separators, key):
        pass

    def draw_evolution_parameter_over_time(self, separators=None, separator_colour=None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination

        Returns:

        '''

        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.groupby_DF(separators)

        # we get back the interesting values, the one that evolves through the chosen dimension:

        # todo change that getter bc not using self
        estimation = self.get_evolution_parameter(self.estimator.DF)
        for key in keys:
            data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            minimum, maximum = self.get_evolution_extremes(data)

            plot.uni_plot(0, estimation, minimum,
                          dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "min",
                                           'marker': ''})
            plot.uni_plot(0, estimation, maximum,
                          dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "max",
                                           'marker': ''})

            # true value line
            true_values = self.get_evolution_true_value(data)
            plot.uni_plot(0, estimation, true_values,
                          dict_plot_param={"color": 'r', "linestyle": "solid", "linewidth": 0.4,
                                           "label": "true value", 'marker': ''})

            # crazy stuff
            if separator_colour is not None:
                estimator = Estimator(data)
                coloured_dict, coloured_keys = estimator.groupby_DF([separator_colour])
                color = plt.cm.Dark2.colors  # Dark2 is qualitative cm and pretty dark cool colors.
                for coloured_key, c in zip(coloured_keys, color):
                    coloured_data = coloured_dict.get_group(coloured_key)
                    coloured_data = self.get_evolution_plot_data(coloured_data)
                    plot.uni_plot(0, estimation, coloured_data,
                                  dict_plot_param={"color": c, "linestyle": "solid", "linewidth": 1.1,
                                                   "label": coloured_key})
            else:
                data = self.get_evolution_plot_data(data)
                plot.uni_plot(0, estimation, data)

            fig_dict = self.get_dict_fig_evolution_parameter_over_time(separators, key)
            plot.set_dict_fig(0, fig_dict)
            plot.show_legend()
            name_file = classical_functions.tuple_to_str(key) + 'evol_estimation'
            plot.save_plot(name_save_file=name_file)