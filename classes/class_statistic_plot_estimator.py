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
from classes import class_histogram_estimator

np.random.seed(124)

# errors:
import errors.Error_convergence
import errors.Warning_deprecated
import errors.Error_forbidden


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Statistic_plot_estimator(Graph_Estimator):
    def __init__(self, estimator, separators=None):
        super().__init__(estimator = estimator, separators = separators)

    @abstractmethod
    def get_computation_plot_fig_dict(self, convergence_in):
        # convergence_in is simply a check parameter. Perhaps we will erase it, but it is actually usefull in graph estimator hawkes.
        pass

    @abstractmethod
    def rescale_time_plot(self, rescale_factor, times):
        pass

    @abstractmethod
    def rescale_sum(self, sum, times):
        '''
        rescale the data, for instance the MSE. The method is useful bc I can rescale with attributes.
        Abstract method allows me to use specific scaling factor.

        :param sum:
        :param times:
        :return:
        '''
        pass

    def convergence_estimators_limit(self, mini_T, times, name_column_evolution, computation_function, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.groupby_DF(separators)

        comp_sum = np.zeros(self.estimator.DF[name_column_evolution].nunique())
        for key in keys:
            data = global_dict.get_group(key)
            estimator = Estimator(data.copy())

            self.test_true_value(data) # test if there is only one true value i  the given sliced data. It could lead to potential big errors.
            estimator.function_upon_separated_data("value", computation_function, "computation",
                                                   true_parameter=estimator.DF["true value"].mean())

            comp_sum += estimator.DF.groupby([name_column_evolution])["computation"].sum()#.values

        TIMES_plot = self.rescale_time_plot(mini_T, times)
        comp_sum = self.rescale_sum(comp_sum, times).values

        plot = APlot()
        plot.uni_plot(0, TIMES_plot, comp_sum, dict_plot_param= {"linewidth" : 2})
        fig_dict = self.get_computation_plot_fig_dict(convergence_in = "MSE")
        plot.set_dict_fig(0, fig_dict)
        plot.save_plot(name_save_file=''.join([computation_function.__name__,'_comput']) )

        #I create a histogram:
        # first, find the DF with only the last estimation, which should always be the max value of column_evolution.
        max_value_evol = self.estimator.DF[name_column_evolution].max()
        hist_DF = self.estimator.DF[ self.estimator.DF[name_column_evolution] == max_value_evol].copy() #copy() for independance

        my_hist = class_histogram_estimator.Histogram_estimator(hist_DF, separators= separators)
        my_hist.draw_histogram()
        return