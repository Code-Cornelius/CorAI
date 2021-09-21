from priv_lib_estimator import Relplot_estimator

from nn_classes.estimator.hyper_parameters.plot_estim_hyper_paramm import Plot_estim_hyper_param


class Relplot_hyper_param(Plot_estim_hyper_param, Relplot_estimator):
    EVOLUTION_COLUMN = None

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estim_hyper_param=estimator, *args, **kwargs)
        return

    def get_dict_fig(self, grouped_data_by, key=None, yscale='linear', **kwargs):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="Comparison of the impact of hyper-parameter on the resulting losses,")
        fig_dict = {'title': title,
                    'xlabel': kwargs['xlabel'],
                    'ylabel': kwargs['ylabel'],
                    'xscale': 'log', 'yscale': 'log',
                    'basex': 10, 'basey': 10}
        return fig_dict

    def scatter(self, column_name_draw, second_column_to_draw_abscissa,
                column_name_true_values=None, separators_plot=None, palette='PuOr',
                hue=None, style=None, markers=None, sizes=None, dict_plot_for_main_line={}, hue_norm=None,
                legend='full', path_save_plot=None, *args, **kwargs):
        kwargs['xlabel'] = second_column_to_draw_abscissa
        kwargs['ylabel'] = column_name_draw
        return super().scatter(column_name_draw, column_name_true_values, separators_plot, palette, hue, style, markers,
                               sizes, dict_plot_for_main_line, hue_norm, legend, second_column_to_draw_abscissa,
                               path_save_plot, *args, **kwargs)
