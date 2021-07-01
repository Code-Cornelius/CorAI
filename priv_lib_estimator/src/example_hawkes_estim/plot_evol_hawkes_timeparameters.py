# normal libraries
import numpy as np

# priv_libraries
from priv_lib_estimator import Relplot_estimator
from priv_lib_estimator.src.example_hawkes_estim.plot_estim_hawkes import Plot_estim_hawkes


# other files


# section ######################################################################
#  #############################################################################
# code:


class Plot_evol_hawkes_timeparameters(Plot_estim_hawkes, Relplot_estimator):
    EVOLUTION_NAME = 'time estimation'
    ESTIMATION_COLUMN_NAME = 'value'
    TRUE_ESTIMATION_COLUMN_NAME = 'true value'

    def __init__(self, estimator_hawkes, fct_parameters, *args, **kwargs):
        super().__init__(estimator_hawkes, fct_parameters, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data

    def get_data2true_evolution(self, data):
        return self.get_evolution_name_specific_data(data, self.TRUE_ESTIMATION_COLUMN_NAME)

    def get_data2evolution(self, data):
        return self.get_evolution_name_specific_data(data, self.ESTIMATION_COLUMN_NAME)

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, separators, key):
        # TODO LOOK AT THE TITLE
        title = self.generate_title(parameters=separators,
                                    parameters_value=key,
                                    before_text="",
                                    extra_text="Only 5-95% of the interval is shown, batches of {} simulations, time: {} until {}",
                                    extra_arguments=[self.nb_of_guesses,
                                                     self.range_estimation[0],
                                                     self.range_estimation[1]])

        fig_dict = {'title': "Evolution of the estimation, " + title,
                    'xlabel': 'Time',
                    'ylabel': "Estimation"}
        return fig_dict

    def draw(self, feature_to_draw, kernels_to_plot=None,
             true_values_flag=False, envelope_flag=True,
             separators_plot=None, separator_colour=None,
             path_save_plot=None,
             *args, **kwargs):

        #                   kernel_plot_param=None, one_kernel_plot_param=None, all_kernels_drawn=False)

        """
        plot the evolution of the estimators over the attribute given by.
        It is almost the same version as the upper class, the difference lies in that I m drawing the kernel on the graph additionally.
        I draw the kernels iff I give kernel_plot_param.

        kernel_plot_param for drawing over a list of kernels, one_kernel_plot for drawing the kernels in the middle.

        Args:
            separators_plot:
            separator_colour: the column of the dataframe to consider for color discrimination
            envelope_flag: list_of_kernels, Times = kernel_plot_param. Used in order to plot all the decided kernels.
        Returns:

        """
        # TODO REWRITE FOR MULTI DIM

        # we use the coloured keys for identifying which colors goes to whom in the one kernel plot case. We assume in the list_of_kernels all name are unique.

        # plot the estimation
        plots, coloured_keys = super().draw(feature_to_draw, true_values_flag, envelope_flag,
                                            separators_plot=separators_plot, separator_colours=separator_colour,
                                            save_plot=False, not_use_grouping_by=False,
                                            *args, **kwargs)
        # on top of the estimation, plot the kernels if given
        if kernels_to_plot is not None:
            list_kernels, list_position_centers = kernels_to_plot
            for plot in plots:
                for kernel, center_time in zip(list_kernels, list_position_centers):
                    tt = [np.linspace(*self.estimator.range_estimation, 3000)]
                    yy = kernel.eval(tt,
                                     center_time)  # TODO FALSE WHAT I AM WRITING HERE MISSING AN ARG, regarding kernel
                    plot.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                         dict_plot_param={"color": "m", "markersize": 0, "linewidth": 0.4,
                                                          "linestyle": "--"})
                    # plot line on the x center of the kernel
                    lim_ = plots.axs[0].get_ylim()
                    plot.plot_vertical_line(center_time, np.linspace(0, lim_[-1] * 0.92, 5), nb_ax=0,
                                            dict_plot_param={"color": "k", "markersize": 0, "linewidth": 0.2,
                                                             "linestyle": "--"})
                if path_save_plot is not None:
                    plot.save_plot(name_save_file=path_save_plot)

# TODO the following is the case where we plot multiple kernels but only once in the middle.
#  It is a very specific case where we compare kernels, not sure if useful anymore.
# elif one_kernel_plot_param is not None:
#     list_of_kernels, Time = one_kernel_plot_param
#
#     # here is all the plots I draw. I start at 1 bc I always plot the parameters as a first drawing.
#     list_of_plots = APlot.print_register()[1:]
#     # on each plot
#     for counter, plots in enumerate(list_of_plots):
#         # for each eval point
#
#         colors = plt.cm.Dark2.colors  # Dark2 is qualitative cm and pretty dark cool colors.
#
#         # we use the coloured keys for identifying which colors goes to whom in the one kernel plot case. We assume in the list_of_kernels all name are unique.
#         for number, (kernel_name, color) in enumerate(zip(coloured_keys, colors)):
#             # basically, we retrieve the name and find the matching kernel.
#             kernel_counter = 0
#             kernel = None
#             while kernel is None and kernel_counter < len(list_of_kernels):
#                 if list_of_kernels[kernel_counter].name == kernel_name:
#                     kernel = list_of_kernels[kernel_counter]
#                 else:
#                     kernel_counter += 1
#             if kernel_counter > len(list_of_kernels):  # if he hasn't found the kernel, there is an error.
#                 raise Exception("The kernels given and plotted are not matching.")
#             tt = [np.linspace(self.T_max * 0.05, self.T_max * 0.95, 3000)]
#             yy = kernel.eval(tt, Time, self.T_max)
#             plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
#                                   dict_plot_param={"color": color, "markersize": 0, "linewidth": 0.7,
#                                                    "linestyle": "--"}, tight=False)
#         # lim_ = plots.axs[0].get_ylim()
#         # plots.plot_vertical_line(Time, np.linspace(0, lim_[-1] * 0.9, 5), nb_ax=0,
#         #                         dict_plot_param={"color": "k", "markersize": 0, "linewidth": 1,
#         #                         "linestyle": "--"})
#         name_file = 'double_estimation_result_{}'.format(counter)
#         plots.save_plot(name_save_file=name_file)
#
# return
