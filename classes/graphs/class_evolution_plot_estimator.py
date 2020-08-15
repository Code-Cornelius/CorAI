# normal libraries
from abc import abstractmethod

import functions.tools.classical_functions_str
import numpy as np  #maths library and arrays
from matplotlib import pyplot as plt  #ploting

# my libraries
from functions.tools import classical_functions
from classes.plot.class_aplot import APlot
from classes.class_estimator import Estimator
from classes.graphs.class_graph_estimator import Graph_Estimator

# errors:

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Evolution_plot_estimator(Graph_Estimator):
    # abstract evolution_name parameter
    @property
    @abstractmethod
    def evolution_name(self):
        pass

    def __init__(self, estimator, separators=None):
        super().__init__(estimator = estimator, separators = separators)

    # section ######################################################################
    #  #############################################################################
    # data

    @classmethod
    @abstractmethod
    def get_evolution_name_unique_values(cls, data):
        pass

    @classmethod
    @abstractmethod
    def get_evolution_name_extremes(cls, data):
        pass

    @abstractmethod
    def get_evolution_name_true_value(self, data):
        pass

    @abstractmethod
    def get_evolution_name_plot_data(self, data):
        pass

    @abstractmethod
    def get_evolution_name_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute, and we focus on data given by column str, computing the means and returning an array.

        Args:
            data:
            str:

        Returns:

        '''
        pass

    # section ######################################################################
    #  #############################################################################
    # plot


    @abstractmethod
    def get_dict_fig(self, separators, key):
        pass

    def draw(self, separators=None, separator_colour=None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination

        Returns:

        '''
        separators, global_dict, keys = super().draw(separators = separators)
        estimation = self.get_evolution_name_unique_values(self.estimator.DF)
        for key in keys:
            data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            minimum, maximum = self.get_evolution_name_extremes(data)

            plot.uni_plot(0, estimation, minimum,
                          dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "min",
                                           'marker': ''})
            plot.uni_plot(0, estimation, maximum,
                          dict_plot_param={"color": 'r', "linestyle": "dashdot", "linewidth": 0.5, "label": "max",
                                           'marker': ''})

            # true value line
            true_values = self.get_evolution_name_true_value(data)
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
                    coloured_data = self.get_evolution_name_plot_data(coloured_data)
                    plot.uni_plot(0, estimation, coloured_data,
                                  dict_plot_param={"color": c, "linestyle": "solid", "linewidth": 1.1,
                                                   "label": coloured_key})
            else:
                data = self.get_evolution_name_plot_data(data)
                plot.uni_plot(0, estimation, data)

            fig_dict = self.get_dict_fig(separators, key)
            plot.set_dict_fig(0, fig_dict)
            plot.show_legend()
            name_file = ''.join([functions.tools.classical_functions_str.tuple_to_str(key), 'evol_estimation'])
            plot.save_plot(name_save_file=name_file)

        # either coloured keys have been defined or not. I retrieve them in order to know what color to put upon which kernel.
        if separator_colour is not None:
            return plot, coloured_keys
        else :
            return plot, None
