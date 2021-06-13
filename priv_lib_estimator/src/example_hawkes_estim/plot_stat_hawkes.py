#normal libraries


#my libraries
from library_classes.estimators.graphs.class_statistic_plot_estimator import *
from library_errors.Error_not_yet_allowed import Error_not_yet_allowed
#other files
from classes.class_estimator_hawkes import *


# batch_estimation is one dataframe with the estimators.
from classes.graphs.class_graph_estimator_hawkes import Graph_Estimator_Hawkes


class Statistic_plot_estimator_Hawkes(Graph_Estimator_Hawkes, Statistic_plot_estimator):

    def __init__(self, estimator, fct_parameters, *args, **kwargs):
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator
        super().__init__(estimator, fct_parameters, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data

    def rescale_time_plot(self, mini_T, times):
        # I multiply by 50 bc I convert the time axis to jump axis, and a mini T corresponds to 50 jumps.
        return [times[i] // mini_T * 50 for i in range(len(times))]

    def rescale_sum(self, my_sum, times):
        """
        rescale the data, for instance the MSE. The method is useful bc I can rescale with attributes.

        Args:
            my_sum:
            times:

        Returns:

        """
        return my_sum / self.nb_of_guesses

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, convergence_in):
        # todo the fig_dict could be more general to adapt to some situations, for now I simply put an if statement.
        if convergence_in == "MSE":
            dict_fig = {
                'title': f"Convergence in MSE sense of the estimators, batches of {self.nb_of_guesses} realisations.",
                'xlabel': "Nb of Events",
                'ylabel': "MSE",
                'parameters': [self.ALPHA[0][0](0, 1, 1), self.BETA[0][0](0, 1, 1), self.NU[0](0, 1, 1)],
                'name_parameters': ["ALPHA", "BETA", "NU"]
            }
            return dict_fig
        else:
            raise Error_not_yet_allowed("MSE has to be given as convergence_in parameter.")
