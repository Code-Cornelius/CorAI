# normal libraries
from abc import abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my libraries
from classes.class_estimator import Estimator
from plot_functions import APlot
import classical_functions


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
        # TODO 23/07/2020 nie_k:  do not use + since it is not optimized. Can be done better.
        for (name, value) in zip(names, values):
            title += ", " + name + " = " + str(value)

        if extra_text is not None:
            title += "\n" + extra_text.format(*extra_arguments)
        return title

    @abstractmethod
    def get_optimal_range_histogram(self, key, mean):
        pass

    @abstractmethod
    def get_dict_plot_param_for_hist(self, key, mean):
        pass

    @abstractmethod
    def get_dict_fig_hist(self, separators, key):
        pass

    @abstractmethod
    def get_evolution_parameter(self, data):
        pass

    @abstractmethod
    def get_evolution_extremes(self, data):
        pass

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
    def get_computation_plot_fig_dict(self):
        pass

    @abstractmethod
    def get_dict_fig_evolution_parameter_over_time(self, separators, key):
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
            name_file =  classical_functions.tuple_to_str(key) + 'histogram'
            plot.save_plot(name_save_file=name_file)

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
    def rescale_time_plot(self, mini_T, times):
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

        #we get back the interesting values, the one that evolves through the chosen dimension:
        estimation = self.get_evolution_parameter(self.estimator.DF)
        for key in keys:
            data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            minimum, maximum = self.get_evolution_extremes(data)

            plot.uni_plot(0, estimation, minimum, dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "min", 'marker':''})
            plot.uni_plot(0, estimation, maximum, dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "max", 'marker':''})

            # true value line
            true_values = self.get_evolution_true_value(data)
            plot.uni_plot(0, estimation, true_values, dict_plot_param= {"color": 'r', "linestyle": "solid", "linewidth": 0.4, "label": "true value", 'marker':''})

            # crazy stuff
            if separator_colour is not None:
                estimator = Estimator(data)
                coloured_dict, coloured_keys = estimator.groupby_DF([separator_colour])
                color = plt.cm.Dark2.colors  #Dark2 is qualitative cm and pretty dark cool colors.
                for coloured_key, c in zip(coloured_keys,color):
                    coloured_data = coloured_dict.get_group(coloured_key)
                    coloured_data = self.get_evolution_plot_data(coloured_data)
                    plot.uni_plot(0, estimation, coloured_data,
                                  dict_plot_param={"color": c, "linestyle": "solid", "linewidth": 0.8, "label": coloured_key})
            else:
                data = self.get_evolution_plot_data(data)
                plot.uni_plot(0, estimation, data)

            fig_dict = self.get_dict_fig_evolution_parameter_over_time(separators, key)
            plot.set_dict_fig(0, fig_dict)
            plot.show_legend()
            name_file =  classical_functions.tuple_to_str(key) + 'evol_estimation'
            plot.save_plot(name_save_file=name_file)

    def convergence_estimators_limit(self, mini_T, times, name_column_evolution, computation_function, separators=None):
        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.groupby_DF(separators)

        comp_sum = np.zeros(self.estimator.DF[name_column_evolution].nunique())
        for key in keys:
            data = global_dict.get_group(key)
            estimator = Estimator(data.copy())

            self.test_true_value(data)
            estimator.function_upon_separated_data("value", computation_function, "computation",
                                                   true_parameter=estimator.DF["true value"].mean())

            comp_sum += estimator.DF.groupby([name_column_evolution])["computation"].sum()#.values

        TIMES_plot = self.rescale_time_plot(mini_T, times)
        comp_sum = self.rescale_sum(comp_sum, times).values

        plot = APlot()
        plot.uni_plot(0, TIMES_plot, comp_sum)
        fig_dict = self.get_computation_plot_fig_dict()
        plot.set_dict_fig(0, fig_dict)
        plot.save_plot(name_save_file='MSE_comput')
        #I create a histogram:
        # first, find the DF with only the last estimation, which should always be the max value of column_evolution.
        max_value_evol = self.estimator.DF[name_column_evolution].max()
        hist_DF = self.estimator.DF[ self.estimator.DF[name_column_evolution] == max_value_evol].copy()

        #BIANCA-HERE this is not good!!
        old_estimator_DF = self.estimator.DF
        self.estimator.DF = hist_DF
        self.draw_histogram()
        self.estimator.DF = old_estimator_DF

    # method for putting higher csv than on DF.
    def to_csv(self, path, **kwargs):
        self.DF.to_csv(path, **kwargs)
        return