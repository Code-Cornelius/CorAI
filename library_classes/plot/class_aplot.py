# normal libraries
import math  # quick math functions
import warnings

import numpy as np  # maths library and arrays
from matplotlib import pyplot as plt  # plotting
import seaborn as sns  # environment for plots

sns.set()  # better layout, like blue background

# my libraries
from library_metaclasses.metaclass_register import *
from library_functions.tools.classical_functions_dict import up

# errors:
from library_errors.Error_not_allowed_input import Error_not_allowed_input


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# plot graph can plot up to 2 graphs on the same figure.
# every argument has to be a list in order to make it work.
# title and labels has to be list, where one has :
# [title 1, title 2] ; [x1label y1label, x2label y2label]
# the set of parameters is the same for the two subplots.

# don't forget to write down #plt.show() at the end !


# section ######################################################################
#  #############################################################################
# new plot functions


class APlot(object, metaclass=register):
    # TODO 23/08/2020 nie_k:  point plot for one point.

    # APlot is the class for my plots. APlot is one figure.

    DEFAULT_DICT_PLOT_PARAMETERS = {"color": 'm',
                                    "linestyle": "solid",
                                    "linewidth": 0.5,
                                    "marker": "o",
                                    "markersize": 0.4,
                                    "label": "plot"
                                    }
    FONTSIZE = 14.5

    @deco_register
    def __init__(self, how=(1, 1), datax=None, datay=None, figsize=(7, 5), sharex=False,
                 sharey=False):  # sharex,y for sharing the same on plots.
        # how should be a tuple with how I want to have axes.
        if datay is not None:
            if datax is not None:
                plt.figure(figsize=figsize)
                plt.plot(datax, datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)
            else:
                plt.figure(figsize=figsize)
                plt.plot(range(len(datay)), datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)

        else:  # corresponds to the case where we want to plot something
            # creation of the figure
            self.fig, self.axs = plt.subplots(*how, sharex=sharex, sharey=sharey, figsize=figsize)
            # true or false uni plot
            self.uni_dim = (how == (1, 1))
            # two cases, if it is uni_dim, I put self.axs into a list. Otherwise, it is already a list.
            # having a list is easier to deal with.
            if self.uni_dim:
                self.axs = [self.axs]
            else:
                # the axs are matrices, I need a list.
                self.axs = self.axs.flatten()
            # now, self.axs is always a list (uni dimensional).
            self.nb_of_axs = how[0] * how[1]  # nb of axes upon which I can plot

            # for the axs_bis, I store the axs inside this guy:
            self.axs_bis = [None] * self.nb_of_axs  # a list full of zeros.

            # we set the default param of the fig:
            for i in range(self.nb_of_axs):
                self.set_dict_fig(i, None)

    def check_axs(self, ax):
        if ax < 0:
            warnings.warn("Axs given is negative. Lists are cyclic.")
        if ax >= self.nb_of_axs:
            warnings.warn("Axs given is out of bounds. I plot upon the first axis.")
            ax = 0
        return ax

    def set_dict_fig(self, nb_ax=0, dict_fig=None, xx=None, yy=None):
        # always plotter first, then dict_updates (using the limits of the axis).
        # dict authorised:
        # {'title', 'xlabel', 'ylabel', 'xscale', 'xint', 'yint','parameters','name_parameters'}
        nb_ax = self.check_axs(nb_ax)
        DEFAULT_STR = "Non-Defined."
        if dict_fig is None:
            dict_fig = {}
        default_dict = {'title': DEFAULT_STR, 'xlabel': DEFAULT_STR, 'ylabel': DEFAULT_STR,
                        'xscale': 'linear', 'yscale': 'linear',
                        'xint': False, 'yint': False}
        up(dict_fig, default_dict)

        self.axs[nb_ax].set_title(dict_fig['title'], fontsize=APlot.FONTSIZE)
        self.axs[nb_ax].set_xlabel(dict_fig['xlabel'], fontsize=APlot.FONTSIZE)
        self.axs[nb_ax].set_ylabel(dict_fig['ylabel'], fontsize=APlot.FONTSIZE)
        self.axs[nb_ax].set_xscale(dict_fig['xscale'])
        self.axs[nb_ax].set_yscale(dict_fig['yscale'])
        self.axs[nb_ax].tick_params(labelsize=APlot.FONTSIZE - 1)

        if dict_fig['xint']:
            if xx is None:
                raise Exception("xx has not been given.")
            x_int = range(math.ceil(min(xx)) - 1, math.ceil(
                self.axs[nb_ax](
                    xx)) + 1)  # I need to use ceil on both if min and self.axs[nb_ax] are not integers ( like 0 to 1 )
            self.axs[nb_ax].set_xticks(x_int)
        if dict_fig['yint']:
            if yy is None:
                raise Exception("yy has not been given.")
            y_int = range(min(yy), math.ceil(self.axs[nb_ax](yy)) + 1)
            self.axs[nb_ax].set_yticks(y_int)

        # I keep the condition. If not true, then no need to move the plot up.
        if 'parameters' in dict_fig and 'name_parameters' in dict_fig:
            parameters = dict_fig['parameters']
            name_parameters = dict_fig['name_parameters']
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

            bottom, top = self.axs[nb_ax].get_ylim()
            left, right = self.axs[nb_ax].get_xlim()
            self.axs[nb_ax].text(left + (right - left) * 0.15, bottom - (top - bottom) * 0.42, sous_text,
                                 fontsize=APlot.FONTSIZE - 1)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace=0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots

    def __my_plotter(self, nb_ax, xx, yy, dict_plot_param, bis=False):
        """
        A helper function to make a graph

        Parameters
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
            nb_ax = self.check_axs(nb_ax)
            if not bis:  # bis is plot on second axis.
                out = self.axs[nb_ax].plot(xx, yy, **dict_plot_param)
                self.axs[nb_ax].grid(True)
            else:
                out = self.axs_bis[nb_ax].plot(xx, yy, **dict_plot_param)
                self.axs[nb_ax].grid(False)
                self.axs_bis[nb_ax].grid(False)
            return out
        else:
            raise Error_not_allowed_input("Inputs for the plot are not of matching size.")


    def uni_plot(self, nb_ax, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig=None, tight=True):
        """
        Method to have 1 plot. Upon nb_ax (int)
        """
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)
        if tight:
            self.fig.tight_layout()
        if dict_fig is not None:
            self.set_dict_fig(nb_ax, dict_fig, xx, yy)

        return

    def uni_plot_ax_bis(self, nb_ax, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                        dict_fig=None, tight=True):
        """ for now I add the ax bis to self.axs at the end. Access through -1.
        """

        #
        if self.axs_bis[nb_ax] is None:  # axis not created yet.
            self.axs_bis[nb_ax] = self.axs[nb_ax].twinx()  # instantiate a second axes that shares the same x-axis
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param, bis=True)

        if tight:
            self.fig.tight_layout()

        if dict_fig is not None:
            self.set_dict_fig(nb_ax, dict_fig, xx, yy)
        return

    def bi_plot(self, nb_ax1, nb_ax2, xx1, yy1, xx2, yy2,
                dict_plot_param_1=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                dict_plot_param_2=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                dict_fig_1=None,
                dict_fig_2=None):
        self.uni_plot(nb_ax1, xx1, yy1, dict_plot_param=dict_plot_param_1, dict_fig=dict_fig_1)
        self.uni_plot(nb_ax2, xx2, yy2, dict_plot_param=dict_plot_param_2, dict_fig=dict_fig_2)
        return

    def plot_function(self, function, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), not_numpy=True):
        # ax is an int, not necessary for uni dim case.
        if not_numpy:
            xx = np.array(xx)
        yy = function(xx)

        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)
        return

    def plot_line(self, a, b, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """
        Plot a line on the chosen ax.

        Args:
            a: slope of line
            b: origin of line
            xx: data, where to have the points of the line
            nb_ax: which ax to use, should be an integer.
            dict_plot_param:  if I want to customize the plot.

        Returns:

        """
        function = lambda x: a * x + b
        return self.plot_function(function, xx, nb_ax=nb_ax, dict_plot_param=dict_plot_param)

    def plot_vertical_line(self, x, yy, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        return self.uni_plot(nb_ax=nb_ax, xx=np.full(len(yy), x), yy=yy, dict_plot_param=dict_plot_param, tight=False)

    def cumulative_plot(self, xx, yy, nb_ax=0):
        """
        add cumulative plot of an nb_axis, for the chosen data set.

        Args:
            xx: xx where points should appear
            yy: the output data.
            nb_ax: which axis.

        Returns:

        """

        ax_bis = self.axs[nb_ax].twinx()
        ax_bis.plot(xx, np.cumsum(yy) / (np.cumsum(yy)[-1]), color='darkorange',
                    marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
        ax_bis.set_ylabel('cumulative ratio')
        ax_bis.set_ylim([0, 1.1])
        self.axs[nb_ax].legend(loc='best')
        return

    default_dict_param_hist = {'bins': 20,
                               "color": 'green', 'range': None,
                               'label': "Histogram", "cumulative": True}

    def hist(self, data, nb_of_ax=0,
             dict_param_hist=default_dict_param_hist.copy(),
             # I need to copy because I am updating it. In particular I pop the cumulative.
             dict_fig=None):

        # function for plotting histograms
        if dict_fig is not None:
            self.set_dict_fig(nb_of_ax, dict_fig)
        self.axs[nb_of_ax].set_xlabel("Realisation")
        self.axs[nb_of_ax].set_ylabel("Nb of realisation inside a bin.")

        up(dict_param_hist, APlot.default_dict_param_hist)

        try:
            # if doesn't pop, it will be catch by except.
            if dict_param_hist.pop("cumulative"):
                values, base, _ = self.axs[nb_of_ax].hist(data, density=False, alpha=0.5, **dict_param_hist)
                ax_bis = self.axs[nb_of_ax].twinx()
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
            values, base, _ = self.axs[nb_of_ax].hist(data, density=False, alpha=0.5, **dict_param_hist)
        return

    def show_legend(self, nb_ax=None):
        # as usually, nb_ax is an integer.
        # if ax is none, then every nb_ax is showing the nb_ax.
        if nb_ax is None:
            for nb_ax_0 in range(self.nb_of_axs):
                self.axs[nb_ax_0].legend(loc='best', fontsize=12)
        else:
            self.axs[nb_ax].legend(loc='best', fontsize=12)
        return

    @staticmethod
    def save_plot(name_save_file='image'):
        """
        Method for saving the plot (figure) created.

        Args:
            name_save_file: name of the file

        Returns: nothing.
        """
        plt.savefig(name_save_file + '.png', dpi=800)
        return
