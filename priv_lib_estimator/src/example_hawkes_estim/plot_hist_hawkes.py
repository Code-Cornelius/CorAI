# normal libraries


# my libraries
from library_classes.estimators.graphs.class_histogram_estimator import *

# other files
from classes.class_estimator_hawkes import *


# batch_estimation is one dataframe with the estimators.
from classes.graphs.class_graph_estimator_hawkes import Graph_Estimator_Hawkes


class Histogram_estimator_Hawkes(Graph_Estimator_Hawkes, Histogram_estimator):
    NB_OF_BINS = 60

    def __init__(self, estimator, fct_parameters, *args, **kwargs):
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator

        super().__init__(estimator, fct_parameters, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # data

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_range(self, key, mean):
        """The best range for parameters is the following. It is then scaled up depending on the mean value.
        Args:
            key:
            mean:

        Returns:

        """
        variable = key[0]
        if variable == "nu":
            return 0.1, 1.5 * mean
        else:
            return 0.6 * mean, 1.4 * mean

    # TODO: make more general -- don't assume that the name will always be the first
    def get_dict_param(self, key, mean):
        my_range = self.get_range(key, mean)
        dict_param = {'bins': Histogram_estimator_Hawkes.NB_OF_BINS,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': my_range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig(self, separators, key):
        title = self.generate_title(names=separators, values=key,
                                    before_text="Histogram for the estimator of a Hawkes Process;",
                                    extra_text="Time of simulation {}", extra_arguments=[self.T_max])
        fig_dict = {'title': title,
                    'xlabel': "Estimation",
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict
