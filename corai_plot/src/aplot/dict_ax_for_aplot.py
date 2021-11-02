# todo add a ylabel bis !
# write down that, if label on bis axis, need to pass the label to the other dict. Be careful bc two axison same figure and so can be crowded.
class Dict_ax_for_APlot(object):
    """
    dict_ax_for_APlot is an object that stores the properties of each axs of a APlot.
    DEFAULT_DICT is then showing the default properties for each axs before personalisation.

    The parameters are:
        title: message on top of the image.
        xlabel: legend of x-axis. string.
        ylabel: legend of y-axis. string.
        xscale: scale of the x-axis. string.
        yscale: scale of the y-axis. string.
        basex: base for log scale on x-axis. float.
        basey: base for log scale on y-axis. float.
        parameters: values of the parameters we want to print under the figure. list of floats. Should not be longer than 20.
        name_parameters: name of the parameters shown next to the value. list of strings. Should not be longer than 20.
        xlim: range of the x-axis. 2 elements list or tuple of floats.
        ylim: range of the y-axis. 2 elements list or tuple of floats.


    """

    # default parameters
    DEFAULT_STR = "Non-Defined."
    DEFAULT_DICT = {'title': DEFAULT_STR,
                    'xlabel': DEFAULT_STR, 'ylabel': DEFAULT_STR,
                    'xscale': 'linear', 'yscale': 'linear',
                    'basex': 10, 'basey': 10,
                    'xint': False, 'yint': False,
                    'parameters': None, 'name_parameters': None,
                    'xlim': None, 'ylim': None}

    DEFAULT_DICT_BIS = {'title': '',
                        'xlabel': '', 'ylabel': 'bis_axis',
                        'xscale': 'linear', 'yscale': 'linear',
                        'basex': 10, 'basey': 10,
                        'xint': False, 'yint': False,
                        'parameters': None, 'name_parameters': None,
                        'xlim': None, 'ylim': None}

    def __init__(self, nb_of_axs):
        # creates a list of independent dicts with the default settings.
        self.list_dicts_parameters_for_each_axs = [Dict_ax_for_APlot.DEFAULT_DICT.copy()
                                                   for _ in range(nb_of_axs)]
        self.list_dicts_parameters_for_each_axs_bis = [Dict_ax_for_APlot.DEFAULT_DICT_BIS.copy()
                                                       for _ in range(nb_of_axs)]

    # TODO it would be a good idea to design the setter with certain conditions:
    # if another parameter than authorised is given, warning!
    # parameters and name_parameters same length.
    # check that scale and xint not set at the same time?

    @classmethod
    def help_dict_ax(cls):
        """
        Semantics:
            print possibilities for dict_ax and the default behavior.
        """
        text = cls.DEFAULT_DICT
        print(text)
