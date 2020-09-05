# normal libraries
import math  # quick math functions
import warnings

import numpy as np  # maths library and arrays
from matplotlib import pyplot as plt  # plotting
import seaborn as sns  # environment for plots

sns.set() #better layout, like blue background

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
## [title 1, title 2] ; [x1label y1label, x2label y2label]
# the set of parameters is the same for the two subplots.

### don't forget to write down #plt.show() at the end !


# section ######################################################################
#  #############################################################################
# new plot functions


class Modularplot(metaclass=register):
    DEFAULT_DICT_PLOT_PARAMETERS = {"color": 'm',
                               "linestyle": "solid",
                               "linewidth": 0.5,
                               "marker": "o",
                               "markersize": 0.4,
                               "label": "plot"
                                    }
    FONTSIZE = 14.5


    @deco_register
    def __init__(self):
        self.upper_ax = plt.subplot2grid((21, 21), (0, 0), rowspan=14,
                                    colspan=21)
        self.lower_ax = plt.subplot2grid((21, 21), (16, 0), rowspan=8, colspan=21)
        self.axs = [self.upper_ax, self.lower_ax]


    def set_dict_fig(self, nb_ax=0, dict_fig=None, xx=None, yy=None):
        # always plotter first, then dict_updates (using the limits of the axis).
        # dict authorised:
        # {'title', 'xlabel', 'ylabel', 'xscale', 'xint', 'yint','parameters','name_parameters'}
        DEFAULT_STR = "Non-Defined."
        if dict_fig is None:
            dict_fig = {}
        default_dict = {'title': DEFAULT_STR, 'xlabel': DEFAULT_STR, 'ylabel': DEFAULT_STR,
                        'xscale': 'linear', 'xint': False, 'yint': False}
        up(dict_fig, default_dict)

        self.axs[nb_ax].set_title(dict_fig['title'], fontsize=Modularplot.FONTSIZE)
        self.axs[nb_ax].set_xlabel(dict_fig['xlabel'], fontsize=Modularplot.FONTSIZE)
        self.axs[nb_ax].set_ylabel(dict_fig['ylabel'], fontsize=Modularplot.FONTSIZE)
        self.axs[nb_ax].set_xscale(dict_fig['xscale'])
        self.axs[nb_ax].tick_params(labelsize=Modularplot.FONTSIZE - 1)

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
                                 fontsize=Modularplot.FONTSIZE - 1)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace=0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots

    def __my_plotter(self, nb_ax, xx, yy, dict_plot_param):
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

        Returns
        -------
        out : list
            list of artists added
        """
        if len(xx) == len(yy):
            up(dict_plot_param, Modularplot.DEFAULT_DICT_PLOT_PARAMETERS)
            out = self.axs[nb_ax].plot(xx, yy, **dict_plot_param)
            self.axs[nb_ax].grid(True)
            return out
        else:
            raise Error_not_allowed_input("Inputs for the plot are not of matching size.")





    def _upper_plot(self, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig=None):
        """
        Method to have 1 plot. Upon nb_ax (int)
        """
        self.__my_plotter(0, xx, yy, dict_plot_param)
        if dict_fig is not None:
            self.set_dict_fig(0, dict_fig, xx, yy)
        return


    def _lower_plot(self, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig=None):
        """
        Method to have 1 plot. Upon nb_ax (int)
        """
        self.__my_plotter(1, xx, yy, dict_plot_param)
        if dict_fig is not None:
            self.set_dict_fig(1, dict_fig, xx, yy)
        return

    def total_plot(self, xx, yy1, yy2, dict_plot_param1=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig1=None,
                   dict_plot_param2=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_fig2=None):
        self._upper_plot(xx, yy1, dict_plot_param=dict_plot_param1, dict_fig=dict_fig1)
        self._lower_plot(xx, yy2, dict_plot_param=dict_plot_param2, dict_fig=dict_fig2)
        return




    def show_legend(self, nb_ax=None):
        # as usually, nb_ax is an integer.
        # if ax is none, then every nb_ax is showing the nb_ax.
        if nb_ax is None:
            for nb_ax_0 in range(1):
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



# one = Modularplot()
# xx = np.linspace(0,1,1000)
# yy1 = xx*xx
# yy2 = xx*xx*xx
# one.total_plot(xx, yy1, yy2, dict_plot_param2 = {"color":"red"})
# plt.show()