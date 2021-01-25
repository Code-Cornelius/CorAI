class APlot_plot_dicts_for_each_axs(object):
    """
    APlot_plot_dicts_for_each_axs is an object that stores the properties of each axs of a APlot.
    default_dict is then showing the default properties for each axs before personalisation.

    The parameters are:
        title: message on top of the image.
        xlabel: legend of x-axis.
        ylabel: legend of y-axis.
        xscale: scale of the x-axis.
        yscale: scale of the y-axis.
        basex: base for lognormal scale on x-axis.
        basey: base for lognormal scale on y-axis.


    """

    #default parameters
    DEFAULT_STR = "Non-Defined."
    default_dict = {'title': DEFAULT_STR,
                    'xlabel': DEFAULT_STR, 'ylabel': DEFAULT_STR,
                    'xscale': 'linear', 'yscale': 'linear',
                    'basex': 10, 'basey': 10,
                    'xint': False, 'yint': False,
                    'parameters' : None, 'name_parameters' : None}

    def __init__(self, nb_of_axs):
        # creates a list of independent dicts with the default settings.
        self.list_dicts_parameters_for_each_axs = [APlot_plot_dicts_for_each_axs.default_dict.copy() for _ in range(nb_of_axs)]

    #TODO it would be a good idea to design the setter with certain conditions:
    # if another parameter than authorised is given, warning!
