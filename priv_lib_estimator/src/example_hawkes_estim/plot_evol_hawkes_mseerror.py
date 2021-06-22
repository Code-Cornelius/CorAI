# normal libraries
import numpy as np

# my libraries
from priv_lib_estimator import Evolution_plot_estimator
from priv_lib_estimator.src.example_hawkes_estim.plot_estim_hawkes import Plot_estim_hawkes


# other files


# section ######################################################################
#  #############################################################################
# code:


class Plot_evol_hawkes_mseerror(Plot_estim_hawkes, Evolution_plot_estimator):
    EVOLUTION_NAME = "length simulation time"
    ESTIMATION_COLUMN_NAME = 'MSE'
    DATA_COLUMN_NAME = 'value'
    TRUE_ESTIMATION_COLUMN_NAME = 'true value'

    def __init__(self, estimator_hawkes, fct_parameters, *args, **kwargs):
        super().__init__(estimator_hawkes, fct_parameters, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################

    def get_data2evolution(self, data):
        return self.get_evolution_name_specific_data(data, Plot_evol_hawkes_mseerror.ESTIMATION_COLUMN_NAME)

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, separators, key):
        title = self.generate_title(parameters=separators,
                                    parameters_value=key,
                                    before_text="",
                                    extra_text="Only 5-95% of the interval is shown, batches of {} simulations, time: {} until {}",
                                    extra_arguments=[self.nb_of_guesses,
                                                     self.range_estimation[0],
                                                     self.range_estimation[1]])
        # true value needs to be given in order to compute MSE and the parameters are called alpha, beta, nu.
        fig_dict = {
            'title': f"Convergence in MSE sense of the estimators, batches of {self.nb_of_guesses} realisations.",
            'xlabel': "Nb of Events",
            'ylabel': self.ESTIMATION_COLUMN_NAME,
            'parameters': [self.ALPHA[0][0](0, 1, 1), self.BETA[0][0](0, 1, 1), self.NU[0](0, 1, 1)],
            'name_parameters': ["ALPHA", "BETA", "NU"]
        }
        return fig_dict

    def draw(self, feature_to_draw, kernels_to_plot=None,
             true_values_flag=False, envelope_flag=True,
             separators_plot=None, separator_colour=None, path_save_plot=None,
             *args, **kwargs):
        super().draw(column_name_draw=self.ESTIMATION_COLUMN_NAME,
                     true_values_flag=False,
                     envelope_flag=False,
                     separators_plot=separators_plot,
                     separator_colour=separator_colour,
                     path_save_plot=path_save_plot,
                     dict_plot_for_main_line={"linewidth": 2},
                     not_use_grouping_by=True,
                     *args, **kwargs)
        # TODO 23/06/2021 nie_k: verify that for each time, unique MSE.

        #### FOR THIS VERY SIMPLE,CREATE A HIST, AND ASK TO DRAW THE MSE COLUMN.
        # filter, and draw.
        # if class_for_hist is not None:
        #     # I create a histogram:
        #     # first, find the DF with only the last estimation, which should always be the max value of column_evolution.
        #     max_value_evol = self.estimator.DF[name_column_evolution].max()
        #
        #     hist_DF = self.estimator.__class__(
        #         self.estimator.DF[self.estimator.DF[name_column_evolution]
        #                           == max_value_evol].copy())  # copy() for independence
        #
        #     my_hist = class_for_hist(hist_DF, *args, **kwargs)
        #     my_hist.draw()
        return

    def compute_mse_along_sep(self, separators):
        # sum up over separators and grouping by.
        def compute_MSE(param, true_parameter):
            return (param - true_parameter) ** 2

        separators, global_dict, keys = super().draw(separators=separators)
        for key in keys:
            data = global_dict.get_group(key)
            # wip data is modified, what about estimator?

            # potentially this is the same for all keys
            values_of_unique_times = self.get_values_evolution_column(data)
            nb_unique_times = len(values_of_unique_times)

            comp_sum = np.zeros(nb_unique_times)
            # wip += here ?
            self.data[self.ESTIMATION_COLUMN_NAME] = self.data.apply(
                lambda row: compute_MSE(row[self.DATA_COLUMN_NAME],
                                        true_parameter=data[self.TRUE_ESTIMATION_COLUMN_NAME].mean()),
                axis=1)
        return
