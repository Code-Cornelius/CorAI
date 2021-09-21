from priv_lib_estimator import Distplot_estimator

from priv_lib_ml.src.nn_classes.estimator.hyper_parameters.plot_estim_hyper_paramm import Plot_estim_hyper_param


class Distplot_hyper_param(Plot_estim_hyper_param, Distplot_estimator):

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estim_hyper_param=estimator, *args, **kwargs)

    def get_dict_fig(self, grouped_data_by, key, **kwargs):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="Histogram of the performances with respect to the hyper-parameters")
        fig_dict = {'title': title,
                    'xlabel': kwargs['column_name_draw'],
                    'ylabel': "Nb of bins"}
        return fig_dict
