import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from priv_lib_ml.src.classes.estimator.history.plot_estim_history import Plot_estim_history

from priv_lib_estimator import Relplot_estimator
from priv_lib_plot import AColorsetContinuous, APlot


class Relplot_history(Plot_estim_history, Relplot_estimator):
    EVOLUTION_COLUMN = 'epoch'

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator=estimator, *args, **kwargs)
        return

    def get_data2evolution(self, data, feature_to_draw):
        return self.get_data2group_sliced(data, feature_to_draw).mean().to_numpy()

    def get_dict_fig(self, grouped_data_by, key=None, yscale='linear', **kwargs):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="")
        fig_dict = {'title': title,
                    'xlabel': self.EVOLUTION_COLUMN,
                    'ylabel': 'Loss',
                    'xscale': 'linear', 'yscale': yscale,
                    'basex': 10, 'basey': 10}
        return fig_dict

    def draw_two_metrics_same_plot(self, key_for_second_axis_plot=None, log_axis_for_loss=True,
                                   log_axis_for_second_axis=False):
        aplot = APlot()

        # adjusting the linewidth depending on nb of plots:
        if self.estimator.nb_folds < 3:
            linewidth = 2
            interval_colors = (0.5, 0.9)  # colormaps
        else:
            linewidth = 1
            interval_colors = (0.3, 1.)

        color_plot_blue = AColorsetContinuous('Blues', self.estimator.nb_folds, interval_colors)
        color_plot_green = AColorsetContinuous('Greens', self.estimator.nb_folds, interval_colors)
        color_plot_red = AColorsetContinuous('Reds', self.estimator.nb_folds, interval_colors)
        color_plot_orange = AColorsetContinuous('Purples', self.estimator.nb_folds, interval_colors)

        if log_axis_for_loss:
            yscale = 'log'
        else:
            yscale = 'linear'

        coloured_dict, coloured_keys = self.estimator.groupby('fold')
        for i, coloured_key in enumerate(coloured_keys):  # iterating over the folds.
            coloured_data = coloured_dict.get_group(coloured_key)
            evolution_xx = self.get_values_evolution_column(coloured_data)

            dict_plot_param_loss_training = {"color": color_plot_green[i],
                                             "linewidth": linewidth,
                                             "label": f"Loss for Training nb {i + 1}"}

            aplot.uni_plot(nb_ax=0, xx=evolution_xx, yy=self.get_data2evolution(coloured_data, 'loss_training'),
                           dict_plot_param=dict_plot_param_loss_training,
                           dict_ax={'title': "Training of a Neural Network, evolution wrt epochs.",
                                    'xlabel': "Epochs", 'ylabel': "Loss",
                                    'xscale': 'linear', 'yscale': yscale,
                                    'basey': 10})
            if key_for_second_axis_plot is not None:
                assert self.estimator.contains(f"{key_for_second_axis_plot}_training"), \
                    f"{key_for_second_axis_plot}_training does not exist in the df"
                dict_plot_param_second_metric_training = {"color": color_plot_blue[i],
                                                          "linewidth": linewidth,
                                                          "label": f"{key_for_second_axis_plot} for Training nb {i + 1}"
                                                          }
                aplot.uni_plot_ax_bis(nb_ax=0, xx=evolution_xx,
                                      yy=self.get_data2evolution(coloured_data, f"{key_for_second_axis_plot}_training"),
                                      dict_plot_param=dict_plot_param_second_metric_training,
                                      dict_ax={'ylabel': key_for_second_axis_plot})

        flag_valid = self.estimator.validation

        if flag_valid:
            if log_axis_for_second_axis:
                dict_ax = {'yscale': 'log'}
            else:
                dict_ax = None

            for i, coloured_key in enumerate(coloured_keys):  # iterating over the folds.
                coloured_data = coloured_dict.get_group(coloured_key)
                evolution_xx = self.get_values_evolution_column(coloured_data)
                dict_plot_param_loss_validation = {"color": color_plot_orange[i],
                                                   "linewidth": linewidth,
                                                   "label": f"Loss for Validation nb {i + 1}"
                                                   }
                aplot.uni_plot(nb_ax=0, xx=evolution_xx, yy=self.get_data2evolution(coloured_data, 'loss_validation'),
                               dict_plot_param=dict_plot_param_loss_validation)
                if key_for_second_axis_plot is not None:
                    assert self.estimator.contains(f"{key_for_second_axis_plot}_validation"), \
                        f"{key_for_second_axis_plot}_training does not exist in the df"
                    dict_plot_param_second_mtrc_val = {"color": color_plot_red[i],
                                                       "linewidth": linewidth,
                                                       "label": f"{key_for_second_axis_plot} for Validation nb {i + 1}"
                                                       }
                    aplot.uni_plot_ax_bis(nb_ax=0, xx=evolution_xx,
                                          yy=self.get_data2evolution(coloured_data,
                                                                     f"{key_for_second_axis_plot}_validation"),
                                          dict_plot_param=dict_plot_param_second_mtrc_val, dict_ax=dict_ax)

        # plot lines of best NN:
        if len(self.estimator.list_best_epoch) > 0:
            _plot_best_epoch_NN(aplot, self.estimator.list_best_epoch)

        aplot.show_legend()
        aplot._axs[0].grid(True)
        if key_for_second_axis_plot is not None:
            aplot._axs_bis[0].grid(True)

        return aplot

    def lineplot(self, log_axis_for_loss=True):
        #### data processing
        name_col_loss = self.estimator.get_col_metric_names()
        df_to_plot = self.estimator.df.melt(id_vars=['epoch', 'fold'], value_vars=name_col_loss,
                                            var_name='loss type', value_name='loss value')
        # slice if loss type contains training
        row_indexes_training = df_to_plot[(df_to_plot['loss type'].str.contains('training'))].index
        row_indexes_validati = df_to_plot.index.difference(row_indexes_training)
        df_to_plot.loc[row_indexes_training, 'data type'] = 'training'
        df_to_plot.loc[row_indexes_validati, 'data type'] = 'validation'
        # and remove the words training and validation from loss type:
        df_to_plot.loc[:, 'loss type'] = df_to_plot.apply(lambda s: s.loc['loss type'].replace('_training', ''), axis=1)
        df_to_plot.loc[:, 'loss type'] = df_to_plot.apply(lambda s: s.loc['loss type'].replace('_validation', ''),
                                                          axis=1)

        ###### plot
        # special color palette where we add another value in order to not have white lines.
        yscale = 'log' if log_axis_for_loss else 'linear'
        # todo self.estimator.df['fold'].nunique() must be constant per metric.
        if self.estimator.df['fold'].nunique() <= 3:
            col_wrap = 1
        elif self.estimator.df['fold'].nunique() >= 10:
            col_wrap = 3
        elif not self.estimator.df['fold'].nunique() % 2:  # case 0, all even number below 10.
            col_wrap = 2
        else:  # cases odd numbers 5,7,9
            col_wrap = 3
        g = sns.relplot(x="epoch", y="loss value", hue="data type", style="loss type",
                        col="fold", kind="line", data=df_to_plot, col_wrap=col_wrap,
                        height=2, aspect=2, facet_kws={'sharex': False, 'sharey': False,
                                                       'legend_out': True})
        # height is the width, aspect is the ratio

        g.fig.tight_layout()
        axs = g.axes.flatten()
        g._legend._loc = 1 # upper right, where there should be nothing.

        if len(self.estimator.list_best_epoch) > 0:
            for i in range(len(axs)):  # plot lines of best NN:
                ylim = np.array(axs[i].get_ylim())
                dict_plot_param = {"color": "black",
                                   "linestyle": "--",
                                   "linewidth": 0.3,
                                   "markersize": 0,
                                   "label": f"Best model for fold nb {i + 1}"
                                   }
                axs[i].plot(np.full(len(ylim), self.estimator.list_best_epoch[i]), ylim, **dict_plot_param)

        g.set(yscale=yscale)
        return


def _plot_best_epoch_NN(aplot, best_epoch_of_NN):
    # use at the end of plotting since we get y_lims.
    yy = np.array(aplot.get_y_lim(nb_ax=0))
    for i in range(len(best_epoch_of_NN)):
        aplot.plot_vertical_line(best_epoch_of_NN[i], yy, nb_ax=0,
                                 dict_plot_param={"color": "black",
                                                  "linestyle": "--",
                                                  "linewidth": 0.3,
                                                  "markersize": 0,
                                                  "label": f"Best model for fold nb {i + 1}"
                                                  })
