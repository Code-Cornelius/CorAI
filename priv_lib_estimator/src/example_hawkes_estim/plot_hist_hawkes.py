# normal libraries


# my libraries

# other files
from priv_lib_estimator import Distplot_estimator
from priv_lib_estimator.src.example_hawkes_estim.plot_estim_hawkes import Plot_estim_hawkes


class Plot_hist_hawkes(Plot_estim_hawkes, Distplot_estimator):
    NB_OF_BINS = 60

    def __init__(self, estimator, fct_parameters, *args, **kwargs):

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
            return 0., 1.5 * mean
        else:
            return 0.6 * mean, 1.4 * mean

    def get_dict_param(self, key, mean):
        my_range = self.get_range(key, mean)
        dict_param = {'bins': Plot_hist_hawkes.NB_OF_BINS,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': my_range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig(self, separators, key):
        title = self.generate_title(parameters=separators, parameters_value=key,
                                    before_text="Histogram for the estimator of a Hawkes Process;",
                                    extra_text="Time of simulation {}", extra_arguments=[self.T_max])
        fig_dict = {'title': title,
                    'xlabel': "Estimation.",
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict
