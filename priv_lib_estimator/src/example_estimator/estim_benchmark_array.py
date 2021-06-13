# normal libraries


# priv libraries
from priv_lib_estimator import Plot_estimator, Evolution_plot_estimator, Histogram_estimator
from priv_lib_estimator.src.estimator.estim_time import Estim_time


# section ######################################################################
#  #############################################################################
# Classes

class Estim_benchmark_array(Estim_time):
    NAMES_COLUMNS = Estim_time.NAMES_COLUMNS.copy()
    NAMES_COLUMNS.update(('Array Size', 'Method'))  # add to the name_columns the specific columns.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Plot_estim_benchmark_array(Plot_estimator):

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)


class Plot_evol_benchmark_array(Plot_estim_benchmark_array, Evolution_plot_estimator):
    EVOLUTION_COLUMN = 'Array Size'

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)

    def get_data2evolution(self, data, feature_to_draw):
        return Plot_evol_benchmark_array.get_data2group_sliced(data,
                                                               feature_to_draw).mean().to_numpy()

    def get_default_dict_fig(self, grouped_data_by, key=None):
        # TODO it would be even better to change the ylabel for something ... I need to think about it
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="Benchmark" )
        fig_dict = {'title': title,
                    'xlabel': self.EVOLUTION_COLUMN,
                    'ylabel': 'Comput. Time',
                    'xscale': 'log', 'yscale': 'log',
                    'basex': 2, 'basey': 10
                    }
        return fig_dict


class Plot_hist_benchmark_array(Plot_estim_benchmark_array, Histogram_estimator):

    NB_OF_BINS = 15

    def ___init__(self, estimator, *args, **kwargs):
        super().__init__(estimator=estimator, *args, **kwargs)

    def get_dict_param(self, key, mean):
        dict_param = {'bins': Plot_hist_benchmark_array.NB_OF_BINS,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': None,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig(self, separators, key):
        title = self.generate_title(parameters=separators, parameters_value=key,
                                    before_text="Histogram for the time for array access",)
        fig_dict = {'title': title,
                    'xlabel': "Time",
                    'ylabel': "Nb of runs inside a bin."}
        return fig_dict