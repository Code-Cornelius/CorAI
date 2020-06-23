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

    def generate_title(self, names, values, extra_text=None, extra_arguments=None):
        title = ""
        for (name, value) in zip(names, values):
            title += ", " + name + " = " + str(value)

        if extra_text is not None:
            title += "\n" + extra_text.format(*extra_arguments)
        return title

    @abstractmethod
    def get_range(self, key, mean):
        pass

    @abstractmethod
    def get_param_info(self, key, mean):
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
            fig_dict = self.get_fig_dict_hist(separators, key)
            plot.hist(data=data, param_dict_hist=param_dict, fig_dict=fig_dict)



    def estimation_hawkes_parameter_over_time(self, separators=None, separator_colour=None):

        if separators is None:
            separators = self.separators

        global_dict, keys = self.estimator.slice_DF(separators)

        for key in keys:
            data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            minimum, maximum, estimation = self.get_extremes(data)

            plot.uni_plot(0, estimation, minimum, param_dict = {"color": 'r', "linestyle": "dashdot","linewidth": 0.5,"label": "min"})
            plot.uni_plot(0, estimation, maximum, param_dict = {"color": 'r', "linestyle": "dashdot","linewidth": 0.5,"label": "max"})

            # true value line
            true_values = self.get_true_values(data)
            plot.uni_plot(0, estimation, true_values, param_dict = {"color": 'r', "linestyle": "solid","linewidth": 0.4,"label": "true value"})

            # crazy stuff
            if separator_colour is not None:
                estimator = Estimator(data)
                coloured_dict, coloured_keys = estimator.slice_DF([separator_colour])
                for coloured_key in coloured_keys:
                    coloured_data = coloured_dict.get_group(coloured_key)
                    coloured_data = self.get_plot_data(coloured_data)
                    # todo colours and labels
                    plot.uni_plot(0, estimation, coloured_data)
            else:
                data = self.get_plot_data(data)
                plot.uni_plot(0, estimation, data)

            fig_dict = self.get_fig_dict_plot(separators, key)
            plot.set_fig_dict(0, fig_dict)
            plot.show_legend()