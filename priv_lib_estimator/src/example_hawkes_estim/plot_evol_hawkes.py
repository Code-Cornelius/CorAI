# normal libraries
import numpy as np

# my libraries
from priv_lib_estimator import Evolution_plot_estimator
from priv_lib_estimator.src.example_hawkes_estim.plot_estim_hawkes import Plot_estim_hawkes
from priv_lib_plot import APlot


# other files


# section ######################################################################
#  #############################################################################
# code:


class Evolution_plot_estimator_Hawkes(Plot_estim_hawkes, Evolution_plot_estimator):
    EVOLUTION_NAME = 'time estimation'
    NB_OF_KERNELS_DRAWN = 14

    def __init__(self, estimator_hawkes, fct_parameters, *args, **kwargs):
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        #  pcq c'est bizarre le truc où j'ai besoin ou pas des fcts evol...
        super().__init__(estimator_hawkes, fct_parameters, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data

    def get_data2true_evolution(self, data):
        return self.get_evolution_name_specific_data(data, 'true value')

    def get_data2evolution(self, data):
        return self.get_evolution_name_specific_data(data, 'value')

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, separators, key):
        title = self.generate_title(names=separators,
                                    values=key,
                                    before_text="",
                                    extra_text="Only 5-95% of the interval is shown, batches of {} simulations, time: 0 until {}",
                                    extra_arguments=[self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation, " + title,
                    'xlabel': 'Time',
                    'ylabel': "Estimation"}
        return fig_dict

    def draw(self, feature_to_draw, kernels_to_plot=None,
             true_values_flag=False, envelope_flag=True,
             separators=None, separator_colour=None,
             save_plot=True, name_file ='image'):

        #                   kernel_plot_param=None, one_kernel_plot_param=None, all_kernels_drawn=False)

        """
        plot the evolution of the estimators over the attribute given by.
        It is almost the same version as the upper class, the difference lies in that I m drawing the kernel on the graph additionally.
        I draw the kernels iff I give kernel_plot_param.

        kernel_plot_param for drawing over a list of kernels, one_kernel_plot for drawing the kernels in the middle.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination
            envelope_flag: list_of_kernels, Times = kernel_plot_param. Used in order to plot all the decided kernels.
        Returns:

        """
        # TODO REWRITE FOR MULTI DIM

        # we use the coloured keys for identifying which colors goes to whom in the one kernel plot case. We assume in the list_of_kernels all name are unique.

        # plot the estimation
        plots, coloured_keys = super().draw(feature_to_draw, true_values_flag, envelope_flag,
                                            separators_plot=separators, separator_colour=separator_colour,
                                            save_plot=False)
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
                plot.save_plot(name_save_file=name_file)


#
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
