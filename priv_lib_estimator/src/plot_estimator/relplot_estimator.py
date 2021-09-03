# normal libraries
from abc import abstractmethod

import seaborn as sns
from priv_lib_estimator.src.plot_estimator.plot_estimator import Plot_estimator
from priv_lib_plot import APlot
# priv_libraries
from priv_lib_util.tools import function_str


# errors:


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Relplot_estimator(Plot_estimator):
    """
    Semantics:
        Abstract class inheriting from Plot_estimator. Relation Plot.
        The purpose is to automatise the plots showing evolution of a feature
            with respect to an other as a time-series.
        The class is showing common behavior for evolution_plot: retrieving data for the time series,
            plotting according to some standards... lineplot / scatterplot.

        EVOLUTION_COLUMN is the parameter with respect to which the feature is evolving. It is usually time or position.
        It is the x-axis of the evolution plot.

        Abstract members:
            EVOLUTION_COLUMN
            get_dict_fig
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

    def __init__(self, estimator, grouping_by=None, **kwargs):
        # args and kwargs for the child super() method. Do not forget them in child classes.
        super().__init__(estimator=estimator, grouping_by=grouping_by, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data processing

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
        """
        return self.get_data2group_sliced(data, column_slice).mean().to_numpy()

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
    def get_dict_fig(self, grouped_data_by, key=None, **kwargs):
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

    def draw(self, separators_plot=None, not_use_grouping_by=False, *args, **kwargs):
        # args and kwargs for the child super() method. Do not forget them in child classes.
        super().draw(*args, **kwargs)
        pass

    def lineplot(self, column_name_draw, column_name_true_values=None, envelope_flag=True, separators_plot=None,
                 palette='PuOr',
                 hue=None, style=None, markers=None, sizes=None,
                 dict_plot_for_main_line={}, path_save_plot=None, list_aplots=None,
                 second_column_to_draw_abscissa = None,
                 *args, **kwargs):
        """
        Semantics:
            Draw the lineplot.
            Issue with seaborn:
                When writing overloading, always draw before using super().
                Otherwise, the legend is not actualised.
                If it is, then seaborn's style in the legend will be erased.

        Args:
            column_name_draw (str): The column of the dataframe used to retrieve the data to draw
            column_name_true_values (str): The column where true values are present
            envelope_flag (bool): Flag to specify whether or not to draw the min and max of the data
            separators_plot (list of str): List of columns to group by, this will generate a plot for the product of unique
                elements in each column
            palette (colors of seaborn): colors for hue. Some example in priv_lib_plot/acolor/colors_seaborn.py
            hue (str): The column used in order to draw multiple lines on the same plot for the data,
                discriminating by this column
            style (str): separator for line and marker style.
            markers (bool):
            sizes:
            dict_plot_for_main_line (dict): additional parameters for the plot (evolution line).
            path_save_plot (str): Path to specify where the plot should be saved. Not saved if is None.
            list_aplots (list of Aplot): Plot for plotting.
                There should be as many axs as in the number of groups given by separators_plotters
                 (with the grouping_by parameter of the plot_estimator).
            second_column_to_draw_abscissa (str): column of the dataframe to use as the abscissa.
            kwargs: passed to super and to get_dict_fig.

        Returns:

        Dependency:
            todo write the dependency

        """
        x_axis = second_column_to_draw_abscissa if not None else self.EVOLUTION_COLUMN
        # super call for gathering all separators together and having the group by done.
        separators_plot, global_dict, keys = super().draw(separators_plot=separators_plot, *args, **kwargs)
        self._raise_if_separator_is_evolution(separators_plot)  # test evolution_name is not part of separators.

        current_plots = list_aplots if list_aplots is not None else []  # fetch the aplots of create an empty list
        current_keys = [] # we return current_keys which is the list version of keys, in the iteration order.
        for i, key in enumerate(keys):
            if key is None:  # case where we cannot use groupby.
                data = global_dict
                separators_plot = ["data used"]  # for the title
                key = ["whole dataset"]  # for the title
            else:
                data = global_dict.get_group(key)
            # TODO 13/07/2021 nie_k:  is key necessarly iterable? what if there is only one key? (i.e. not 0 or 3...?)
            current_keys.append(key)

            # creation of the plot
            # choice of ax
            if list_aplots is not None:
                # TODO 27/06/2021 nie_k:  verify plots is same length as keys.
                aplot = list_aplots[i]
                ax = aplot._axs[i]
            else:
                aplot = APlot()
                current_plots.append(aplot)
                ax = aplot._axs[0]
            sns.lineplot(x=x_axis, y=column_name_draw,
                         hue=hue, style=style, sizes=sizes, markers=markers,
                         legend='full', ci=95, err_style="band",
                         palette=palette,
                         data=data, ax=ax,  **dict_plot_for_main_line)

            if envelope_flag:
                for fct in ['min', 'max']:
                    sns.lineplot(x=x_axis, y=column_name_draw,
                                 estimator=fct, hue=hue,
                                 legend=False, err_style="band", ci=None,
                                 # no Conf. Inter. for max value (does not make actually sense with bootstrapping)
                                 palette=palette, data=data, ax=ax,
                                 color='r', linestyle='--', linewidth=0.5, label=fct)
            if column_name_true_values is not None:
                sns.lineplot(x=x_axis, y=column_name_true_values,
                             hue=hue, legend=False, err_style="band", ci=None,
                             palette=palette, data=data, ax=ax,
                             color='r', linestyle='--', linewidth=0.5, label='true value')

            # TODO 01/07/2021 nie_k:  is there a way to retrieve the complete legend, with annotations?
            fig_dict = self.get_dict_fig(separators_plot, key, **kwargs)
            aplot.set_dict_ax(0, fig_dict)

            super()._saveplot(aplot, path_save_plot, 'relplot_', key)
        return current_plots, current_keys

    def scatter(self, column_name_draw, column_name_true_values=None, separators_plot=None,
                palette='PuOr',
                hue=None, style=None, markers=None, sizes=None,
                dict_plot_for_main_line={}, hue_norm = None, legend='full',
                second_column_to_draw_abscissa = None,
                path_save_plot=None,
                *args, **kwargs):
        # TODO 27/06/2021 nie_k:  add the ax parameter as it is in line plot.
        """
        Semantics:
            Draw the scatterplot.
            Issue with seaborn:
                When writing overloading, always draw before using super().
                Otherwise, the legend is not actualised.
                If it is, then seaborn's style in the legend will be erased.

        Args:
            column_name_draw (str): The column of the dataframe used to retrieve the data to draw
            column_name_true_values (str): The column where true values are present
            separators_plot (list of str): List of columns to group by, this will generate a plot for the product of unique
                elements in each column
            palette (colors of seaborn): colors for hue. Some example in priv_lib_plot/acolor/colors_seaborn.py
            hue (str): The column used in order to draw multiple lines on the same plot for the data,
                discriminating by this column
            style (str): separator for marker style
            markers (bool):
            sizes:
            dict_plot_for_main_line (dict): additional parameters for the plot (evolution line).
            hue_norm (tuple):
            second_column_to_draw_abscissa (str): column of the dataframe to use as the abscissa.
            path_save_plot (str): Path to specify where the plot should be saved. Not saved if is None.
            kwargs:
        Returns:

        Dependency:
            todo write the dependency
        """
        x_axis = second_column_to_draw_abscissa if not None else self.EVOLUTION_COLUMN

        # super call for gathering all separators together and having the group by done.
        separators_plot, global_dict, keys = super().draw(separators_plot=separators_plot, *args, **kwargs)
        self._raise_if_separator_is_evolution(separators_plot)  # test evolution_name is not part of separators.
        plots = []
        for key in keys:
            if key is None:  # case where we cannot use groupby.
                data = global_dict
                separators_plot = ["data used"]  # for title
                key = ["whole dataset"]  # for title
            else:
                data = global_dict.get_group(key)
            # creation of the plot
            plot = APlot()
            plots.append(plot)

            sns.scatterplot(x=x_axis, y=column_name_draw,
                            hue=hue, style=style, sizes=sizes, markers=markers,
                            legend=legend, palette=palette, hue_norm = hue_norm,
                            data=data, ax=plot._axs[0], **dict_plot_for_main_line)

            if column_name_true_values is not None:
                sns.lineplot(x=x_axis, y=column_name_true_values,
                             hue=hue,
                             legend=False, palette=palette, data=data, ax=plot._axs[0],
                             color='r', linestyle='--', linewidth=0.5, label='true value',
                             **dict_plot_for_main_line)

            fig_dict = self.get_dict_fig(separators_plot, key, **kwargs)
            plot.set_dict_ax(0, fig_dict)

            if path_save_plot is not None:
                # TODO 23/06/2021 nie_k:  PROBLEM WHEN KEY IS NONE Le chemin d’accès spécifié est introuvable: ''
                # TODO 26/06/2021 nie_k: refactor function using path_save_plot + fct. LIKE PARAMETER + KEYS + NAME FCT (hist, scatter, lineplot...)
                name_file = ''.join([function_str.tuple_to_str(key, ''), 'evol_estimation'])
                plot.save_plot(name_save_file=name_file)
        return plots

    # section ######################################################################
    #  #############################################################################
    # testing

    def _raise_if_separator_is_evolution(self, grouping_by):
        # test evolution_name is not part of separators.
        if self.EVOLUTION_COLUMN in grouping_by:
            raise ValueError("One cannot put the EVOLUTION_COLUMN inside the separators.")
        return
