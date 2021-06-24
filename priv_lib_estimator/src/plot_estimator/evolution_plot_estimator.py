# normal libraries
from abc import abstractmethod

# my libraries
from priv_lib_util.tools import function_str
from priv_lib_estimator.src.plot_estimator.plot_estimator import Plot_estimator
from priv_lib_plot import APlot

# errors:


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Evolution_plot_estimator(Plot_estimator):
    """
    Semantics:
        Abstract class inheriting from Plot_estimator.
        The purpose is to automatise the plots showing evolution of a feature
            with respect to an other as a time-series.
        The class is showing common behavior for evolution_plot: retrieving data for the time series,
            plotting according to some standards...

        EVOLUTION_COLUMN is the parameter with respect to which the feature is evolving. It is usually time or position.
        It is the x-axis of the evolution plot.

        Abstract members:
            EVOLUTION_COLUMN
            get_default_dict_fig
            get_data2evolution

            if you want to use true_evolution you will want to redefine:
                get_data2true_evolution

    """

    # abstract EVOLUTION_COLUMN parameter
    @property
    @abstractmethod
    def EVOLUTION_COLUMN(self):
        # EVOLUTION_COLUMN is a string, column name from the estimator.
        pass

    def __init__(self, estimator, grouping_by=None, *args, **kwargs):
        super().__init__(estimator=estimator, grouping_by=grouping_by, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data processing

    @abstractmethod
    def get_data2evolution(self, data, column_slice):
        """
        Semantics:
            Retrieve the data for the evolution.
            First groupby the evolution feature,
            then slice the interested data.

        Args:
            data:
            column_slice: the data we are interested in. Should be a list of key.

        Returns:

        Examples for child class:
            def get_evolution_name_plot_data(self, data):
                return self.get_grouped_evolution_name_feature(data, 'value').mean().to_numpy()
        """
        pass

    def get_data2true_evolution(self, data, column_slice):
        """

        Args:
            data:
            column_slice (list(str)): the data we are interested in. Should be a list of key.

        Returns:


        Examples for child class:
            def get_evolution_name_true_value(self, data):
                return self.get_grouped_evolution_name_feature(data, 'true value').mean().to_numpy()
        """
        raise ValueError("get_evolution_name_true_value was not implemented in lower class.")

    @classmethod
    def get_data2group_sliced(cls, data, column_slice):
        """
        Semantics:
            Retrieve the series of a Dataframe,
            by grouping according to EVOLUTION_COLUMN and then looking
            at the column "column_slice".


        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.
            column_slice: the data we are interested in. Should be a list of key.

        Returns:
            returns a groupby of the size: (nb groups, nb of features).

        Examples:
            This function can be used by creating two methods in the child class called:
                    def get_evolution_name_true_value(self, data):
                        return self.get_data2group_sliced(data, 'true value').mean().to_numpy()

                    def get_evolution_name_plot_data(self, data):
                        return self.get_data2group_sliced(data, 'value').mean().to_numpy()

        """
        return data.groupby([cls.EVOLUTION_COLUMN])[column_slice]

    @classmethod
    def get_values_evolution_column(cls, data):
        """
        Semantics:
            Retrieve the list of unique values in the column of the EVOLUTION_COLUMN feature of data.
            The list is not sorted, organised in order of appearance.

        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.

        Returns:
            numpy array of the unique values taken in the column EVOLUTION_COLUMN.

        """
        return data[cls.EVOLUTION_COLUMN].unique()

    @classmethod
    def get_evolution_name_extremes(cls, data, features):
        """
        Semantics:
            Retrieve the extremes of a Dataframe,
            when the data is groupby the EVOLUTION_COLUMN parameter,
            and we look at the column feature.

        Args:
            data: data is a pd.DataFrame.
            There is the possibility of not using the estimator as the data.
            features: the features we are interested in. It should be a list of keys for features.

        Returns:
            tuple of two numpy array of the min and max, each numpy array is
            of size: (nb groups, nb of features).

        Dependencies:
            get_data2group_sliced
        """
        values = cls.get_data2group_sliced(data, features)
        return values.min().to_numpy(), values.max().to_numpy()

    # section ######################################################################
    #  #############################################################################
    # plot:

    @abstractmethod
    def get_default_dict_fig(self, grouped_data_by, key=None):
        """
        Semantics:
            default parameters for the drawing of evolution_plot_estimator, depending on grouped_data_by.

        Args:
            grouped_data_by: features we groupby the data with.
            key: the value of the features we groupby-ied with and that we are currently plotting.

        Returns:
            the default dict for evolution_plot_estimator.

        """
        pass

    def draw(self, column_name_draw, true_values_flag=False,
             envelope_flag=True, separators_plot=None,
             separator_colour=None, dict_plot_for_main_line={}, path_save_plot=None,
             *args, **kwargs):
        # TODO 23/06/2021 nie_k: todo the check we did in separator plot, do the same in separator colour.
        #  I want separator color to be a list, not just a string.
        #  It is easy to do as grouping by allows for list of str.
        #  Be ccareful that unpacking a string is the characters.
        """
        Semantics:
            Draw the evolution_plot_estimator common behavior.

        Args:
            column_name_draw (str): The column of the dataframe used to retrieve the data to draw
            true_values_flag (bool): Flag to specify whether or not true values are present and should be drawn
            envelope_flag (bool): Flag to specify whether or not to draw the min and max of the data
            separators_plot (list of str): List of columns to group by, this will generate a plot for the product of unique
                elements in each column
            separator_colour (str): The column used in order to draw multiple lines on the same plot for the data,
                discriminating by this column
            dict_plot_for_main_line (dict): additional parameters for the plot (evolution line).
            path_save_plot (str): Path to specify where the plot should be saved. Not saved if is None.

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
        # TODO 23/06/2021 nie_k: I am sure we made some checks somewhere about
        #  separators are in columns (important condition).
        # here we could do this again, for both separators and colors.

        separators_plot, global_dict, keys = super().draw(separators_plot=separators_plot, *args, **kwargs)
        self._raise_if_separator_is_evolution(separators_plot)  # test evolution_name is not part of separators.

        plots = []
        for key in keys:
            if key is None:  # case where we cannot use groupby.
                data = global_dict
                separators_plot = ["data used"]
                key = ["whole dataset"]
            else:
                data = global_dict.get_group(key)
            plot = APlot()
            plots.append(plot)

            # allow groups to have different ranges for xx.
            # This arr contains all unique values, and so covers all cases from coloured key.
            evolution_xx = self.get_values_evolution_column(data)

            # min and max
            if envelope_flag:
                self._plot_min_max(data, evolution_xx, column_name_draw, plot)
            # true value line
            if true_values_flag:
                self._plot_true_value(data, evolution_xx, plot)

            # discriminating wrt another column. The discrimination will look as different lines plotted.
            if separator_colour is None:
                data = self.get_data2evolution(data, column_name_draw)
                plot.uni_plot(0, evolution_xx, data, dict_plot_param=dict_plot_for_main_line)
            else:  # separator colour given
                coloured_dict, coloured_keys = self.estimator.groupby_data(data, separator_colour)
                # : groupby the data and retrieve the keys.

                if len(coloured_keys) > len(self.COLORMAP):
                    warnings.warn("There is more data than colors, there might be an issue while plotting (not all curves plotted).")

                for coloured_key, c in zip(coloured_keys, self.COLORMAP):
                    coloured_data = coloured_dict.get_group(coloured_key)
                    evolution_xx = self.get_values_evolution_column(coloured_data)
                    # : allow groups to have different ranges for xx.
                    # : If each coloured_data has difference ranges, it is important!

                    coloured_data = self.get_data2evolution(coloured_data, column_name_draw)

                    dict_for_plot = {"color": c, "linestyle": "solid", "linewidth": 1.1,
                                     "label": coloured_key}
                    dict_for_plot.update(dict_plot_for_main_line)

                    plot.uni_plot(0, evolution_xx, coloured_data, dict_plot_param=dict_for_plot)

            self._plot_finalisation(key, plot, path_save_plot, separators_plot)

        # either coloured keys have been defined or not. I retrieve them in order to know what color to put upon which kernel.
        if separator_colour is not None:
            return plots, coloured_keys
        else:
            return plots, None

    def _plot_finalisation(self, key, plot, path_save_plot, separators):
        fig_dict = self.get_default_dict_fig(separators, key)
        plot.set_dict_ax(nb_ax=0, dict_ax=fig_dict, bis_y_axis=False)
        plot.show_legend()
        # TODO 23/06/2021 nie_k:  PROBLEM WHEN KEY IS NONE Le chemin d’accès spécifié est introuvable: ''
        # PROBLEM IN THE LOGIC
        if path_save_plot is not None:
            name_file = ''.join([function_str.tuple_to_str(key, ''), 'evol_estimation'])
            plot.save_plot(name_save_file=name_file)

    def _plot_true_value(self, data, estimation, plot):
        true_values = self.get_data2true_evolution(data)
        plot.uni_plot(0, estimation, true_values, dict_plot_param={"color": 'r', "linestyle": "solid", "linewidth": 0.4,
                                                                   "label": "true value", 'marker': ''})

    def _plot_min_max(self, data, estimation, feature_to_draw, plot):
        minimum, maximum = self.get_evolution_name_extremes(data, feature_to_draw)
        plot.uni_plot(0, estimation, minimum, dict_plot_param={"color": 'r', "linestyle": "dashdot",
                                                               "linewidth": 0.5, "label": "min",
                                                               'marker': ''})
        plot.uni_plot(0, estimation, maximum, dict_plot_param={"color": 'r', "linestyle": "dashdot",
                                                               "linewidth": 0.5, "label": "max",
                                                               'marker': ''})

    # section ######################################################################
    #  #############################################################################
    # testing

    def _raise_if_separator_is_evolution(self, grouping_by):
        # test evolution_name is not part of separators.
        if self.EVOLUTION_COLUMN in grouping_by:
            raise ValueError("One cannot put the EVOLUTION_COLUMN inside the separators.")
        return
