# normal libraries


# priv libraries
import seaborn as sns
from priv_lib_error import Error_type_setter
from priv_lib_estimator import Plot_estimator, Relplot_estimator, Distplot_estimator
from priv_lib_estimator.src.estimator.estim_time import Estim_time


# section ######################################################################
#  #############################################################################
# Classes
from priv_lib_plot import APlot


class Estim_benchmark_array(Estim_time):
    CORE_COL = Estim_time.CORE_COL.copy()
    CORE_COL.update(('Array Size', 'Method'))  # add to the name_columns the specific columns.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Plot_estim_benchmark_array(Plot_estimator):
    def __init__(self, estimator_bench, *args, **kwargs):
        if not isinstance(estimator_bench, Estim_benchmark_array):
            raise Error_type_setter(f'Argument is not an {str(Estim_benchmark_array)}.')
        super().__init__(estimator_bench, *args, **kwargs)


class Relplot_benchmark_array(Plot_estim_benchmark_array, Relplot_estimator):
    EVOLUTION_COLUMN = 'Array Size'

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)
        return

    def get_dict_fig(self, grouped_data_by, key=None, **kwargs):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="Benchmark")
        fig_dict = {'title': title,
                    'xlabel': self.EVOLUTION_COLUMN,
                    'ylabel': 'Comput. Time',
                    'xscale': 'log', 'yscale': 'log',
                    'basex': 2, 'basey': 10
                    }
        return fig_dict

    def lineplot(self, column_name_draw, column_name_true_values=None, envelope_flag=True, separators_plot=None,
                 palette='PuOr',
                 hue=None, style=None, markers=None, sizes=None,
                 dict_plot_for_main_line={}, path_save_plot=None,
                 list_aplots=None,
                 *args, **kwargs):
        list_aplots = [APlot()]
        xx = self.get_values_evolution_column(self.estimator.df)
        list_unique_length = self.get_data2evolution(self.estimator.df, column_name_draw)
        rescaled_xx = xx / xx[0] * list_unique_length[0] * 3
        # rescaled_xx2 = rescaled_xx * rescaled_xx

        print(list_unique_length)
        list_aplots[0].uni_plot(0,xx, rescaled_xx, dict_plot_param={'label':'o(x)', 'linewidth': 0.9,
                                                                          'markersize': 0, 'color': 'black'})
        # list_aplots[0].uni_plot(0, xx, rescaled_xx2, dict_plot_param={'label': 'o(x^2)', 'linewidth'  : 0.9,
        #                                                                             'markersize' : 0, 'color' : 'black'})

        current_plots, keys = super().lineplot(column_name_draw, column_name_true_values, envelope_flag, separators_plot,
                         palette, hue, style, markers, sizes, dict_plot_for_main_line, path_save_plot, list_aplots,
                         *args, **kwargs)
        return current_plots, keys

class Distplot_benchmark_array(Plot_estim_benchmark_array, Distplot_estimator):
    def ___init__(self, estimator, *args, **kwargs):
        super().__init__(estimator=estimator, *args, **kwargs)

    def get_dict_fig(self, separators, key):
        title = self.generate_title(parameters=separators, parameters_value=key,
                                    before_text="Histogram for the time for array access")
        fig_dict = {'title': title,
                    'xlabel': "Time",
                    'ylabel': "Nb of runs inside a bin."}
        return fig_dict

    @staticmethod
    def color_scheme(palette):
        palette = sns.color_palette(palette, n_colors=4)[:3]
        return palette  # only adapted to the particular main
