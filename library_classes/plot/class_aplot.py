# normal libraries
import math  # quick math functions
import warnings

import numpy as np  # maths library and arrays
from matplotlib import pyplot as plt  # plotting
import seaborn as sns  # environment for plots

# my libraries
from library_metaclasses.metaclass_register import *
from library_functions.tools.classical_functions_dict import up
from library_functions.tools.classical_functions_iterable import is_a_container

# errors:
from library_errors.error_not_allowed_input import Error_not_allowed_input

# other files


sns.set()  # better layout, like blue background

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Examples:




"""


# plot graph can plot up to 2 graphs on the same figure.
# every argument has to be a list in order to make it work.
# title and labels has to be list, where one has :
# [title 1, title 2] ; [x1label y1label, x2label y2label]
# the set of parameters is the same for the two subplots.

# don't forget to write down #plt.show() at the end !


class APlot_List_of_dicts_of_parameters(object):
    """
    Class for the list of dicts of parameters used in APlot.
    """
    DEFAULT_STR = "Non-Defined."
    default_dict = {'title': DEFAULT_STR, 'xlabel': DEFAULT_STR, 'ylabel': DEFAULT_STR,
                    'xscale': 'linear', 'yscale': 'linear', 'basex': 10, 'basey': 10,
                    'xint': False, 'yint': False}

    # other accepted parameters:
    # parameters
    # name_parameters
    def __init__(self, nb_of_axs):
        # creates a list of independent dicts with the default settings.
        self.list_dicts_parameters = [APlot_List_of_dicts_of_parameters.default_dict.copy() for _ in range(nb_of_axs)]


# section ######################################################################
#  #############################################################################
# new plot functions


class APlot(object, metaclass=register):
    """
    SEMANTICS : APlot shall be one figure from matplotlib.

    DEPENDENCIES : SEABORN is installed with this class.
    REFERENCES : matplolib.pyplot heavily relied upon.

    The class is linked to register metaclass. This allows to keep track of all the APlot created.
    """

    # class parameter about default plot parameters.
    DEFAULT_DICT_PLOT_PARAMETERS = {"color": 'm',
                                    "linestyle": "solid",
                                    "linewidth": 0.5,
                                    "marker": "o",
                                    "markersize": 0.4,
                                    "label": "plot"
                                    }

    DEFAULT_DICT_PARAM_HIST = {'bins': 20,
                               "color": 'green',
                               'range': None,
                               'label': "Histogram",
                               "cumulative": True}

    # class parameter for fontsize on plots.
    FONTSIZE = 14.5

    @deco_register
    def __init__(self, how=(1, 1), datax=None, datay=None, figsize=(7, 5), sharex=False,
                 sharey=False):
        """
        If datax/datay is not None plot directly. Otherwise, create a figure.

        Args:
            how: should be a tuple with how I want to have axes. e.g. (1,1)
            datax:
            datay: If only datay given, returns a plot wrt nb data.
            figsize: size of the figure.
            sharex: for sharing the same X axis on plots.
            sharey: for sharing the same Y axis on plots.
        """
        if datay is not None:
            if datax is not None:
                plt.figure(figsize=figsize)
                plt.plot(datax, datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)
            else:
                plt.figure(figsize=figsize)
                plt.plot(range(len(datay)), datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)

        else:
            # corresponds to the case where we want to plot something
            # creation of the figure
            self._fig, self._axs = plt.subplots(*how, sharex=sharex, sharey=sharey, figsize=figsize)
            # true or false uni plot
            self._uni_dim = (how == (1, 1))
            # two cases, if it is uni_dim, I put self.axs into a list. Otherwise, it is already a list.
            # having a list is easier to deal with.
            if self._uni_dim:
                self._axs = [self._axs]
            else:
                # the axs are matrices, I need a list.
                self._axs = self._axs.flatten()
            # now, self.axs is always a list (uni dimensional).
            self._nb_of_axs = how[0] * how[1]  # nb of axes upon which I can plot

            # for the axs_bis, I store the axs inside this guy:
            self._axs_bis = [None] * self._nb_of_axs  # a list full of zeros.

            # we set the default param of the fig:
            self.list_dicts_fig_param = APlot_List_of_dicts_of_parameters(
                nb_of_axs=self._nb_of_axs)  # it is a list of dicts.

            # Each element gives the config for the corresponding axes
            # through a dictionary defined by default in the relevant class.
            # it fixes the default behaviour, and the dicts are updated by the user later on.
            for i in range(self._nb_of_axs):
                self.set_dict_fig(i, self.list_dicts_fig_param.list_dicts_parameters[i])

    def __check_axs(self, ax):
        """
        SEMANTICS : verifies the access to axes "ax".
        If negative or bigger than the number of axes of the fig, we warn the user.

        Args:
            ax: unsigned integer

        Returns:

        """
        if ax < 0:
            warnings.warn("Axs given is negative. Lists are cyclic.")
        if ax >= self._nb_of_axs:
            warnings.warn("Axs given is out of bounds. I plot upon the first axis.")
            ax = 0
        return ax

    def __my_plotter(self, nb_ax, xx, yy, dict_plot_param, bis=False):
        """
        SEMANTICS : A helper function to make a graph
        REFERENCES : I took the function from the matplotlib lib, I kept the same name.

        Args
        ----------
        nb_ax : Axes
            The axes to draw upon. Has to be an integer.

        xx : array
           The x data

        yy : array
           The y data

        dict_plot_param : dict
           Dictionary of kwargs to pass to ax.plot

        bis : bool
            if bis draw on bis plot.

        Returns the plot

        """
        if len(xx) == len(yy):
            up(dict_plot_param, APlot.DEFAULT_DICT_PLOT_PARAMETERS)
            nb_ax = self.__check_axs(nb_ax)
            if not bis:  # bis is plot on second axis.
                out = self._axs[nb_ax].plot(xx, yy, **dict_plot_param)
                self._axs[nb_ax].grid(True)
            else:
                out = self._axs_bis[nb_ax].plot(xx, yy, **dict_plot_param)
                self._axs[nb_ax].grid(False)
                self._axs_bis[nb_ax].grid(False)
            return out
        else:
            raise Error_not_allowed_input("Inputs for the plot are not of matching size.")

    def set_dict_fig(self, nb_ax=0, dict_fig=None, xx=None, yy=None):
        """
        SEMANTICS : set some of the figure's characteristics.
        PRECONDITIONS : the parameters allowed are the one
        written in the class APlot_List_of_dicts_of_parameters.

        DEPENDENCIES : The class APlot_List_of_dicts_of_parameters is
        the one that creates the list of dicts used by APlot.

        Args:
            nb_ax: which axs is changed.
            dict_fig: the parameters for config.
            xx: data, for example in the case of changing the ticks.
            yy: data, for example in the case of changing the ticks.

        Returns: void.

        """
        if dict_fig is None:  # case where no need to update the dicts. It would not do anything. So we return
            return

        nb_ax = self.__check_axs(nb_ax)
        dict_param = self.list_dicts_fig_param.list_dicts_parameters[nb_ax]
        # update the default dict with the passed parameters.
        # It changes self.list_dicts_fig_param.list_dicts_parameters[nb_ax].
        up(dict_param, dict_fig)

        self._axs[nb_ax].set_title(dict_param['title'],
                                   fontsize=APlot.FONTSIZE)
        self._axs[nb_ax].set_xlabel(dict_param['xlabel'],
                                    fontsize=APlot.FONTSIZE)
        self._axs[nb_ax].set_ylabel(dict_param['ylabel'],
                                    fontsize=APlot.FONTSIZE)

        # we split the log case, for the possibility of setting up a base. Other cases don't work if you give a base.
        if dict_param['xscale'] == 'log':
            self._axs[nb_ax].set_xscale(dict_param['xscale'],
                                        base=dict_param['basex'])
        else:
            self._axs[nb_ax].set_xscale(dict_param['xscale'])
        if dict_param['xscale'] == 'log':
            self._axs[nb_ax].set_yscale(dict_param['yscale'],
                                        base=dict_param['basey'])
        else:
            self._axs[nb_ax].set_yscale(dict_param['yscale'])

        self._axs[nb_ax].tick_params(labelsize=APlot.FONTSIZE - 1)

        if dict_param['xint']:
            if xx is None:
                raise Exception("xx has not been given.")
            x_int = range(math.ceil(min(xx)) - 1, math.ceil(
                self._axs[nb_ax](
                    xx)) + 1)  # I need to use ceil on both if min and self.axs[nb_ax] are not integers ( like 0 to 1 )
            self._axs[nb_ax].set_xticks(x_int)
        if dict_param['yint']:
            if yy is None:
                raise Exception("yy has not been given.")
            y_int = range(min(yy), math.ceil(self._axs[nb_ax](yy)) + 1)
            self._axs[nb_ax].set_yticks(y_int)

        # I keep the condition. If not true, then no need to move the plot up.
        if 'parameters' in dict_param \
                and 'name_parameters' in dict_param:
            parameters = dict_param['parameters']
            name_parameters = dict_param['name_parameters']
            nb_parameters = len(parameters)
            sous_text = " Parameters : \n"
            for i in range(nb_parameters):
                sous_text += str(name_parameters[i]) + f" = {parameters[i]}"
                # end of the list, we finish by a full stop.
                if i == nb_parameters - 1:
                    sous_text += "."
                # certain chosen number of parameters by line, globally, 3 by line.
                # There shouldn't be more than 16 parameters
                elif i in [4, 7, 10, 13, 16]:
                    sous_text += ", \n "
                # otherwise, just keep writing on the same line.
                else:
                    sous_text += ", "

            bottom, top = self._axs[nb_ax].get_ylim()
            left, right = self._axs[nb_ax].get_xlim()
            self._axs[nb_ax].text(left + (right - left) * 0.15, bottom - (top - bottom) * 0.42, sous_text,
                                  fontsize=APlot.FONTSIZE - 1)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace=0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots
        return

    def show_legend(self, nb_ax=None):
        # as usually, nb_ax is an integer.
        # if ax is none, then every nb_ax is showing the nb_ax.
        if nb_ax is None:
            for nb_ax_0 in range(self._nb_of_axs):
                self._axs[nb_ax_0].legend(loc='best', fontsize=APlot.FONTSIZE - 3)
        else:
            self._axs[nb_ax].legend(loc='best', fontsize=APlot.FONTSIZE - 3)
        return

    @staticmethod
    def show_plot():
        """
        SEMANTICS : adapter for the show pyplot function

        Returns:

        """
        plt.show()
        return

    @staticmethod
    def save_plot(name_save_file='image'):
        """
        SEMANTICS : Method for saving the plot (figure) created.

        Args:
            name_save_file: name of the file

        Returns: nothing.
        """
        plt.savefig(name_save_file + '.png', dpi=800)
        return

    # section ######################################################################
    #  #############################################################################
    # PLOT FUNCTIONS

    def uni_plot(self, nb_ax, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig=None, tight=True):
        """
        SEMANTICS : plot a single plot upon an axis.

        Args:
            nb_ax:
            xx:
            yy:
            dict_plot_param:
            dict_fig:
            tight:

        Returns:

        """
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)
        if tight:
            self._fig.tight_layout()
        if dict_fig is not None:
            self.set_dict_fig(nb_ax, dict_fig, xx, yy)

        return

    def bi_plot(self, nb_ax1, nb_ax2, xx1, yy1, xx2, yy2,
                dict_plot_param_1=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                dict_plot_param_2=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                dict_fig_1=None,
                dict_fig_2=None):
        """
        SEMANTICS : Creates two plot at once.

        DEPENDENCIES : uni_plot

        Args:
            nb_ax1:
            nb_ax2:
            xx1:
            yy1:
            xx2:
            yy2:
            dict_plot_param_1:
            dict_plot_param_2:
            dict_fig_1:
            dict_fig_2:

        Returns:

        """
        self.uni_plot(nb_ax1, xx1, yy1, dict_plot_param=dict_plot_param_1, dict_fig=dict_fig_1)
        self.uni_plot(nb_ax2, xx2, yy2, dict_plot_param=dict_plot_param_2, dict_fig=dict_fig_2)
        return

    def uni_plot_ax_bis(self, nb_ax, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                        dict_fig=None, tight=True):
        """

        Args:
            nb_ax:
            xx:
            yy:
            dict_plot_param:
            dict_fig:
            tight:

        Returns:

        """
        """ for now I add the ax bis to self.axs at the end. Access through -1.
        """

        #
        if self._axs_bis[nb_ax] is None:  # axis not created yet.
            self._axs_bis[nb_ax] = self._axs[nb_ax].twinx()  # instantiate a second axes that shares the same x-axis
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param, bis=True)

        if tight:
            self._fig.tight_layout()

        if dict_fig is not None:
            self.set_dict_fig(nb_ax, dict_fig, xx, yy)
        return

    def cumulative_plot(self, xx, yy, nb_ax=0):
        """
        SEMANTICS : Draw the cumulative distribution of the data yy, with points for the line laying at xx.

        Args:
            xx: xx where points should appear on the graph.
            yy: the data for the cumulative distribution. Real numbers.
            nb_ax: on which axis the plot is drawn.

        Returns:

        """

        ax_bis = self._axs[nb_ax].twinx()
        ax_bis.plot(xx, np.cumsum(np.abs(yy)) / (np.cumsum(np.abs(yy))[-1]), color='darkorange',
                    # the abs function is for making sure that it works even for negative values.
                    marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
        ax_bis.set_ylabel('cumulative ratio')
        ax_bis.set_ylim([0, 1.1])
        self._axs[nb_ax].legend(loc='best')
        return

    def hist(self, data, nb_of_ax=0,
             dict_param_hist=DEFAULT_DICT_PARAM_HIST.copy(),
             dict_fig=None):  # I need to copy because I am updating it.
        """
        SEMANTICS : plotting histograms
        Args:
            data:
            nb_of_ax:
            dict_param_hist:
            dict_fig:

        Returns:

        """
        if dict_fig is not None:
            self.set_dict_fig(nb_of_ax, dict_fig)
        self._axs[nb_of_ax].set_xlabel("Realisation")
        self._axs[nb_of_ax].set_ylabel("Nb of realisation inside a bin.")

        up(dict_param_hist, APlot.DEFAULT_DICT_PARAM_HIST)

        try:
            # if doesn't pop, it will be catch by except.
            if dict_param_hist.pop("cumulative"):
                values, base, _ = self._axs[nb_of_ax].hist(data, density=False, alpha=0.5, **dict_param_hist)
                ax_bis = self._axs[nb_of_ax].twinx()
                values = np.append(values, 0)
                # I add 0 because I want to create the last line, which does not go up.
                # I put then 0 in order to have no evolution with cumsum.

                if 'total_number_of_simulations' in dict_param_hist:
                    ax_bis.plot(base, np.cumsum(values) / dict_param_hist['total_number_of_simulations'],
                                color='darkorange', marker='o',
                                linestyle='-',
                                markersize=1, label="Cumulative Histogram")
                else:
                    ax_bis.plot(base, np.cumsum(values) / np.cumsum(values)[-1],
                                color='darkorange', marker='o', linestyle='-',
                                markersize=1, label="Cumulative Histogram")
                ax_bis.set_ylabel("Proportion of the cumulative total.")

        except KeyError:  # no cumulative in the hist.
            values, base, _ = self._axs[nb_of_ax].hist(data, density=False, alpha=0.5, **dict_param_hist)
        return

    def plot_vertical_line(self, x, yy, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        return self.uni_plot(nb_ax=nb_ax, xx=np.full(len(yy), x), yy=yy, dict_plot_param=dict_plot_param, tight=False)

    def plot_function(self, function, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), not_numpy=True):
        # ax is an int, not necessary for uni dim case.
        if not_numpy:
            xx = np.array(xx)
        yy = function(xx)

        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)
        return

    def plot_line(self, a, b, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """
        SEMANTICS : Plot a line on the chosen ax.

        Args:
            a: slope of line
            b: origin of line
            xx: data, where to have the points of the line
            nb_ax: which ax to use, should be an integer.
            dict_plot_param:  if I want to customize the plot.

        Returns:

        """
        if is_a_container(a) or is_a_container(b):  # are a and b scalars?
            raise Error_not_allowed_input("a and b should be scalars, but containers were given.", a, b)

        function = lambda x: a * x + b
        return self.plot_function(function, xx, nb_ax=nb_ax, dict_plot_param=dict_plot_param)

    def plot_point(self, x, y, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """ plots a single point at (x,y).
        CONDITIONS : plot_point uses plot_line.

        Args:
            x:
            y:
            nb_ax:
            dict_plot_param:

        Returns:

        """
        return self.plot_line(a=0, b=y, xx=x, nb_ax=nb_ax, dict_plot_param=dict_plot_param)
