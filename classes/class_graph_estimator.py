# normal libraries
from abc import abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my libraries
from classes.class_estimator import Estimator
from plot_functions import APlot

class Graph_Estimator:

    def __init__(self, estimator, separators=None):
        self.estimator = estimator
        self.separators = separators

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        estimator = Estimator(pd.read_csv(path))
        # get the max value which is M-1
        return cls(estimator)

    def generate_title(self, names, values, before_text = "", extra_text=None, extra_arguments=[]): # extra_argument is empty list that isn't used.
        title = before_text
        for (name, value) in zip(names, values):
            title += ", " + name + " = " + str(value)

        if extra_text is not None:
            title += "\n" + extra_text.format(*extra_arguments)
        return title

    @abstractmethod
    def get_optimal_range_histogram(self, key, mean):
        pass

    @abstractmethod
    def get_dict_param_for_plot(self, key, mean):
        pass

    @abstractmethod
    def get_fig_dict_hist(self, separators, key):
        pass

    @abstractmethod
    def get_extremes(self, data):
        pass

    @abstractmethod
    def get_true_values(self, data):
        pass

    @abstractmethod
    def get_plot_data(self, data):
        pass

    @abstractmethod
    def get_fig_dict_plot(self, separators, key):
        pass

    def draw_histogram(self, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.slice_DF(separators)

        for key in keys:
            data = global_dict.get_group(key)['value']
            mean = data.mean()
            data = data.values
            plot = APlot()
            param_dict = self.get_dict_param_for_plot(key, mean)
            fig_dict = self.get_fig_dict_hist(separators, key)
            plot.hist(data=data, param_dict_hist=param_dict, fig_dict=fig_dict)



    def draw_evolution_parameter_over_time(self, separators=None, separator_colour=None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.

        Args:
            separators:
            separator_colour:

        Returns:

        '''

        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.slice_DF(separators)

        for key in keys:
            data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            minimum, maximum, estimation = self.get_extremes(data)

            plot.uni_plot(0, estimation, minimum, param_dict={"color": 'r', "linestyle": "dashdot","linewidth": 0.5,"label": "min"})
            plot.uni_plot(0, estimation, maximum, param_dict={"color": 'r', "linestyle": "dashdot","linewidth": 0.5,"label": "max"})

            # true value line
            true_values = self.get_true_values(data)
            plot.uni_plot(0, estimation, true_values, param_dict = {"color": 'r', "linestyle": "solid","linewidth": 0.4,"label": "true value"})

            # crazy stuff
            if separator_colour is not None:
                estimator = Estimator(data)
                coloured_dict, coloured_keys = estimator.slice_DF([separator_colour])
                color = plt.cm.Dark2.colors  #np.linspace(0, 1, len(coloured_keys))))
                for coloured_key, c in zip(coloured_keys,color):
                    coloured_data = coloured_dict.get_group(coloured_key)
                    coloured_data = self.get_plot_data(coloured_data)
                    # todo colours and labels
                    plot.uni_plot(0, estimation, coloured_data,
                                  param_dict={"color": c, "linestyle": "solid","linewidth": 0.8,"label": coloured_key})
            else:
                data = self.get_plot_data(data)
                plot.uni_plot(0, estimation, data)

            fig_dict = self.get_fig_dict_plot(separators, key)
            plot.set_fig_dict(0, fig_dict)
            plot.show_legend()

    def test_true_value(self, data):
        '''
        test if there is only one true value i  the given sliced data.
        It could lead to potential big errors.

        Args:
            data: sliced data from estimator.DF

        Returns:

        '''
        if data['true value'].nunique() != 1:
            raise ("Error because you are estimating different parameters, but still compounding the MSE error together.")

    @abstractmethod
    def get_times_plot(self, mini_T, times):
        pass

    @abstractmethod
    def rescale_sum(self, sum, times):
        '''
        rescale the data depending on the nb_of_guesses. Useful for some properties of the estimator.

        Args:
            sum:
            times:

        Returns:

        '''
        pass

    @abstractmethod
    def get_computation_plot_fig_dict(self):
        pass

    def convergence_estimators_limit(self, mini_T, times, name_column_evolution, computation_function, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.slice_DF(separators)

        comp_sum = np.zeros(self.estimator.DF[name_column_evolution].nunique())
        for key in keys:
            data = global_dict.get_group(key)
            estimator = Estimator(data.copy())

            self.test_true_value(data)
            estimator.function_upon_separated_data("value", computation_function, "computation",
                                                   true_parameter=estimator.DF["true value"].mean())

            comp_sum += estimator.DF.groupby([name_column_evolution])["computation"].sum()#.values

        TIMES_plot = self.get_times_plot(mini_T, times)
        comp_sum = self.rescale_sum(comp_sum, times).values

        plot = APlot()
        plot.uni_plot(0, TIMES_plot, comp_sum)
        fig_dict = self.get_computation_plot_fig_dict()
        plot.set_fig_dict(0, fig_dict)

        #I create a histogram:
        # first, find the DF with only the last estimation, which should always be the max value of column_evolution.
        max_value_evol = self.estimator.DF[name_column_evolution].max()
        hist_DF = self.estimator.DF[ self.estimator.DF[name_column_evolution] == max_value_evol].copy()

        #BIANCA-HERE this is not good!!
        old_estimator_DF = self.estimator.DF
        self.estimator.DF = hist_DF
        self.draw_histogram()
        self.estimator.DF = old_estimator_DF