# normal libraries
from abc import abstractmethod
from matplotlib import pyplot as plt  # plotting
import numpy as np  # maths library and arrays

# my libraries
from priv_lib_util.tools import function_str
from priv_lib_plot import APlot, AColorsetDiscrete
from priv_lib_estimator.src.estimator.estimator import Estimator
from priv_lib_estimator.src.plot_estimator.plot_estimator import Plot_estimator


# errors:


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Evolution_plot_estimator(Plot_estimator):
    """
    Semantics:
        abstract class inheriting from Plot_estimator.
        The purpose is to automatise the plots showing evolution of a feature
        with respect to an other as a time-series.
        The class is showing common behavior for evolution_plot: retrieving data for the time series,
        plotting according to some standards...

        EVOLUTION_NAME is the parameter with respect to which the feature is evolving. It is usually time or position.

        Abstract members:
            EVOLUTION_NAME
            get_default_dict_fig

    """

    # abstract evolution_name parameter
    @property
    @abstractmethod
    def EVOLUTION_NAME(self):
        # EVOLUTION_NAME is a string.
        pass

    def __init__(self, estimator, grouping_by=None, *args, **kwargs):
        super().__init__(estimator=estimator, grouping_by=grouping_by,
                         *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data processing

    @abstractmethod
    def get_evolution_name_plot_data(self, data, feature_to_draw):
        pass

    def get_evolution_name_true_value(self, data, feature_to_draw):
        raise ValueError("get_evolution_name_true_value was not implemented by lower class")

    @classmethod
    def get_grouped_evolution_name_feature(cls, data, features):
        """
        Semantics:
            Retrieve the series of a Dataframe,
            by grouping according to evolution_name and then looking
            at the column feature.


        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.
            features: the features we are interested in. Should be a list of feature keys.

        Returns:
            returns a groupby of the size: (nb groups, nb of features).

        Examples:
            This function can be used by creating two methods in the child class called:
                    def get_evolution_name_true_value(self, data):
                        return self.get_grouped_evolution_name_feature(data, 'true value').mean().to_numpy()

                    def get_evolution_name_plot_data(self, data):
                        return self.get_grouped_evolution_name_feature(data, 'value').mean().to_numpy()

        """
        return data.groupby([cls.EVOLUTION_NAME])[features]

    @classmethod
    def get_evolution_name_unique_values(cls, data):
        """
        Semantics:
            Retrieve the list of unique values in the column of the EVOLUTION_NAME feature of data.
            The list is not sorted, organised in order of appearance.

        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.

        Returns:
            numpy array of the unique values taken in the column EVOLUTION_NAME..

        """
        return data[cls.EVOLUTION_NAME].unique()

    @classmethod
    def get_evolution_name_extremes(cls, data, features):
        """
        Semantics:
            Retrieve the extremes of a Dataframe,
            when the data is groupby the EVOLUTION_NAME parameter,
            and we look at the column feature.


        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.
            features: the features we are interested in. It should be a list of keys for features.

        Returns:
            tuple of two numpy array of the min and max, each numpy array is
            of size: (nb groups, nb of features).

        Dependencies:
            get_grouped_evolution_name_feature
        """
        values = cls.get_grouped_evolution_name_feature(data, features)
        return values.min().to_numpy(), values.max().to_numpy()

    # section ######################################################################
    #  #############################################################################
    # plot:  the logic is that we might want to do multiple plot.
    # So we offer the possibility
    # of having a parameter that split the data (grouped_data_by)
    # and key to know with which data's slice we are working with.

    @abstractmethod
    def get_default_dict_fig(self, grouped_data_by, key=None):
        """
        Semantics:
            default parameters for the drawing of evolution_plot_estimator.

        Args:
            grouped_data_by: features we groupby the data with.
            key: the value of the features we groupbyied with and that we are currently plotting.

        Returns:
            the default dict for evolution_plot_estimator.

        """
        pass

    def draw(self, feature_to_draw, true_values_flag=False, envelope_flag=True,
             separators=None, separator_colour=None):
        """
        Semantics:
            Draw the evolution_plot_estimator common behavior.
    #TODO
        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination

        Returns:

        Dependency:
            get_evolution_name_unique_values

            if envelope_flag:
                get_evolution_name_extremes
            if true_values_flag:
                get_evolution_name_true_value

            get_evolution_name_plot_data
            get_default_dict_fig
        """
        separators, global_dict, keys = super().draw(separators=separators)
        estimation = self.get_evolution_name_unique_values(self.estimator.df)
        for key in keys:
            if key is None:
                data = global_dict
            else:
                data = global_dict.get_group(key)
            plot = APlot()

            # min and max
            if envelope_flag:
                minimum, maximum = self.get_evolution_name_extremes(data, feature_to_draw)
                plot.uni_plot(0, estimation, minimum, dict_plot_param={"color": 'r', "linestyle": "dashdot",
                                                                       "linewidth": 0.5, "label": "min",
                                                                       'marker': ''})
                plot.uni_plot(0, estimation, maximum, dict_plot_param={"color": 'r', "linestyle": "dashdot",
                                                                       "linewidth": 0.5, "label": "max",
                                                                       'marker': ''})
            # true value line
            if true_values_flag:
                true_values = self.get_evolution_name_true_value(data)
                plot.uni_plot(0, estimation, true_values,
                              dict_plot_param={"color": 'r', "linestyle": "solid", "linewidth": 0.4,
                                               "label": "true value", 'marker': ''})

            # discriminating wrt another data
            if separator_colour is None:
                data = self.get_evolution_name_plot_data(data, feature_to_draw)
                plot.uni_plot(0, estimation, data)
            else:
                estimator = Estimator(data)
                coloured_dict, coloured_keys = estimator.groupby_DF([separator_colour])

                for coloured_key, c in zip(coloured_keys, self.COLORMAP):
                    coloured_data = coloured_dict.get_group(coloured_key)
                    coloured_data = self.get_evolution_name_plot_data(coloured_data, feature_to_draw)
                    plot.uni_plot(0, estimation, coloured_data,
                                  dict_plot_param={"color": c, "linestyle": "solid", "linewidth": 1.1,
                                                   "label": coloured_key})

            fig_dict = self.get_default_dict_fig(separators, key)
            plot.set_dict_ax(nb_ax=0, dict_ax=fig_dict, bis_y_axis=False)
            plot.show_legend()
            name_file = ''.join([function_str.tuple_to_str(key, ''), 'evol_estimation'])
            plot.save_plot(name_save_file=name_file)

        # either coloured keys have been defined or not. I retrieve them in order to know what color to put upon which kernel.
        if separator_colour is not None:
            return plot, coloured_keys
        else:
            return plot, None
