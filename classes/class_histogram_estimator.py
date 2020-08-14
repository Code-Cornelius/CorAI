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
import classes.class_estimator
from classes.class_graph_estimator import Graph_Estimator

# errors:
import errors.Error_convergence
import errors.Warning_deprecated
import errors.Error_forbidden

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Histogram_estimator(Graph_Estimator):
    def __init__(self, estimator, separators=None):
        super().__init__(estimator = estimator, separators = separators)

    #todo not abstract but gives back None or whatsoever default.
    # I can't not redefine it if it s not abstract, but I want to be able to not write it down if there is no optimal range.
    @abstractmethod
    def get_optimal_range_histogram(self, key, mean):
        pass

    @abstractmethod
    def get_dict_plot_param_for_hist(self, key, mean):
        pass

    @abstractmethod
    def get_dict_fig_hist(self, separators, key):
        pass

    def draw_histogram(self, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.groupby_DF(separators)

        for key in keys:
            data = global_dict.get_group(key)['value']
            mean = data.mean()
            data = data.values
            plot = APlot()
            param_dict = self.get_dict_plot_param_for_hist(key, mean)
            fig_dict = self.get_dict_fig_hist(separators, key)
            plot.hist(data=data, dict_param_hist=param_dict, dict_fig=fig_dict)
            name_file =  ''.join([classical_functions.tuple_to_str(key), 'histogram'])
            plot.save_plot(name_save_file=name_file)
