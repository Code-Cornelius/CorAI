# normal libraries


# priv libraries
from priv_lib_estimator import Plot_estimator, Evolution_plot_estimator
from priv_lib_estimator.src.estimator.estim_time import Estim_time


# section ######################################################################
#  #############################################################################
# Classes

class Estim_benchmark_array(Estim_time):
    NAMES_COLUMNS = Estim_time.NAMES_COLUMNS.copy()
    NAMES_COLUMNS.update('Array Size', 'Method')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Plot_estim_benchmark_array(Plot_estimator):

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)


class Plot_evol_benchmark_array(Plot_estim_benchmark_array, Evolution_plot_estimator):
    EVOLUTION_NAME = 'Array Size'

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)

    def get_evolution_name_plot_data(self, data, feature_to_draw):
        return Plot_evol_benchmark_array.get_grouped_evolution_name_feature(data,
                                                                            feature_to_draw).mean().to_numpy()

    def get_default_dict_fig(self, grouped_data_by, key):
        # TODO it would be even better to change the ylabel for something ... I need to think about it
        fig_dict = {'title': "Benchmark index & access vs enumerate & element",
                    'xlabel': self.EVOLUTION_NAME,
                    'ylabel': 'Comput. Time'}
        return fig_dict
