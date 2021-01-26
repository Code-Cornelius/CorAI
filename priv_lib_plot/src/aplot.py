# normal libraries
import math  # quick math functions
import warnings

import numpy as np  # maths library and arrays
from matplotlib import pyplot as plt  # plotting
import seaborn as sns  # environment for plots

# my libraries
from priv_lib_metaclass import Register, deco_register
from priv_lib_util.tools import function_dict, function_iterable
from priv_lib_plot.src.aplot_plot_dicts_for_each_axs import APlot_plot_dicts_for_each_axs

# errors:
from priv_lib_error import Error_not_allowed_input

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

# don't forget to write down #show_plot() at the end !


# section ######################################################################
#  #############################################################################
# new plot functions


class APlot(object, metaclass=Register):
    """
    SEMANTICS:
        APlot is an object representing one figure from matplotlib.pyplot.
        The aim is to reduce code duplication by having a standard presentation of plots for 2D plots.
        A figure is cut into multiple subplots that are customizable.

        For now are supported the features:
            -multiple subplots
            -same axis for a few curves
            -duplicating an axis (example: have scales for the same plot)


        Recipe:
            1) create an object APlot with the desired properties.
            2) plot the objects on each axs.
            3) use show_plot or save_plot.


    DEPENDENCIES:
        SEABORN is imported and set with this class.

    REFERENCES:
        matplolib.pyplot heavily relied upon.
        APlot_plot_dicts_for_each_axs which stores the parameters for each axs.

    The class is linked to register metaclass. This allows to keep track of all the APlots created.
    """

    # class parameter about default plot parameters.
    DEFAULT_DICT_PLOT_PARAMETERS = {"color": 'm',
                                    "linestyle": "solid",
                                    "linewidth": 0.5,
                                    "marker": "o",
                                    "markersize": 0.4,
                                    "label": "plot"
                                    }

    DEFAULT_DICT_HIST_PARAMETERS = {'bins': 20,
                                    "color": 'green',
                                    'range': None,
                                    'label': "Histogram",
                                    "cumulative": True}

    # class parameter for fontsize on plots.
    FONTSIZE = 14.5

    @deco_register
    def __init__(self, how=(1, 1),
                 datax=None, datay=None,
                 figsize=(7, 5),
                 sharex=False, sharey=False):
        """
        If datay (and potentially datax) is given, APlot plots directly. Allows quick plotting.

        Args:
            how: should be a tuple with how the axes are organised on the figure e.g. (1,1)
            datax: if given with datay, plot the figure datax, datay.
            datay: If only datay given, returns a plot with respect to range(len(datay)).
            figsize: size of the figure.
            sharex: for sharing the same X axis for two axes. This can be used for having two plots on the same column sharing the same X-axis.
            sharey: for sharing the same Y axis for two axes. This can be used for having two plots on the same line sharing the same Y-axis.
        """

        # quick plot
        if datay is not None:
            if datax is not None:
                plt.figure(figsize=figsize)
                plt.plot(datax, datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)
            else:
                plt.figure(figsize=figsize)
                plt.plot(range(len(datay)), datay, **APlot.DEFAULT_DICT_PLOT_PARAMETERS)

        else:
            # personalised plotting
            self._fig, self._axs = plt.subplots(*how, sharex=sharex, sharey=sharey, figsize=figsize)

            self._uni_dim = (how == (1, 1))  # type boolean

            # two cases, if it is uni_dim, I put self.axs into a list. Otherwise, it is already a list.
            # having a list is easier to deal with.
            if self._uni_dim:
                self._axs = [self._axs]
            else:
                # the axs are matrices that we flatten for simplicity.
                self._axs = self._axs.flatten()

            # now, self.axs is always a list (uni dimensional).
            self._nb_of_axs = how[0] * how[1]  # nb of axes

            # for the axs_bis, I store the axs inside this guy:
            self._axs_bis = [None] * self._nb_of_axs  # a list full of zeros.

            # we set the default param of the fig:
            self.aplot_plot_dicts_for_each_axs = APlot_plot_dicts_for_each_axs(
                nb_of_axs=self._nb_of_axs)  # it is a list of dicts.

            # Each element gives the config for the corresponding axes
            # through a dictionary defined by default in the relevant class.
            # it fixes the default behaviour, and the dicts are updated by the user later on.
            for i in range(self._nb_of_axs):
                self.set_dict_ax(nb_ax = i, dict_ax = self.aplot_plot_dicts_for_each_axs.list_dicts_parameters_for_each_axs[i], bis = False)

    def __check_axs(self, nb_ax):
        """
        SEMANTICS:
            verifies the access to axes "ax".
            If given number negative or bigger than the number of axes of the fig, we warn the user.

        Args:
            nb_ax: unsigned integer

        Returns:

        """
        assert round(nb_ax) == nb_ax, "ax has to be an integer"

        if nb_ax < 0:
            warnings.warn("Axs given is negative. Lists are cyclic.")
        if nb_ax >= self._nb_of_axs:
            warnings.warn("Axs given is out of bounds. I plot upon the first axis.")
            nb_ax = 0
        return nb_ax

    def __my_plotter(self, nb_ax, xx, yy, dict_plot_param, bis=False):
        """
        SEMANTICS :
            A helper function to make a graph.
        REFERENCES :
            The function comes from the matplotlib lib, same name.

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

        Returns:
            plot.

        Examples:
            dict_plot_param can be any dict with these keys (non-exhaustive):
                -label : string
                -color : 'b','g','r','c','m','y','k','w'
                -marker : '.',',','o','v','^','<','>','+','x'
                -markersize : float
                -linestyle : '-','--','-.',':'
                -linewidth : float
            complete list at https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

        """
        if len(xx) == len(yy):
            function_dict.up(dict_plot_param, APlot.DEFAULT_DICT_PLOT_PARAMETERS)
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

    def set_dict_ax(self, nb_ax=0, dict_ax=None, xx=None, yy=None, bis = False):
        """
        SEMANTICS:
            set the parameters of an axes.

        PRECONDITIONS:
            the parameters that can be chosen are the one
            written in the class APlot_plot_dicts_for_each_axs.

        DEPENDENCIES:
            The class APlot_plot_dicts_for_each_axs is
            the one that creates the list of dicts used by APlot.

        Args:
            nb_ax: integer, which axs is changed.
            dict_ax: dictionary with the parameters for configuration.
            xx: one can data, for example in the case of changing the ticks.
            yy: data, for example in the case of changing the ticks.

        Returns:
            void.
        """
        if dict_ax is None:  # case where no need to update the dicts.
            return

        nb_ax = self.__check_axs(nb_ax)
        if bis:
            axis = self._axs_bis[nb_ax]
        else:
            axis = self._axs[nb_ax]

        dict_parameters_for_the_ax = self.aplot_plot_dicts_for_each_axs.list_dicts_parameters_for_each_axs[nb_ax]
        # update the default dict with the passed parameters.
        # It changes self.aplot_plot_dicts_for_each_axs.list_dicts_parameters_for_each_axs[nb_ax].
        dict_parameters_for_the_ax.update(dict_ax)


        axis.set_title(dict_parameters_for_the_ax['title'],
                                   fontsize=APlot.FONTSIZE)
        axis.set_xlabel(dict_parameters_for_the_ax['xlabel'],
                                    fontsize=APlot.FONTSIZE)
        axis.set_ylabel(dict_parameters_for_the_ax['ylabel'],
                                    fontsize=APlot.FONTSIZE)

        # we split the log case, for the possibility of setting up a base.
        # Giving a base without giving the logscale does nothing.
        if dict_parameters_for_the_ax['xscale'] == 'log':
            axis.set_xscale(dict_parameters_for_the_ax['xscale'],
                                        base=dict_parameters_for_the_ax['basex'])
        else:
            axis.set_xscale(dict_parameters_for_the_ax['xscale'])

        if dict_parameters_for_the_ax['yscale'] == 'log':
            axis.set_yscale(dict_parameters_for_the_ax['yscale'],
                                        base=dict_parameters_for_the_ax['basey'])
        else:
            axis.set_yscale(dict_parameters_for_the_ax['yscale'])

        axis.tick_params(labelsize=APlot.FONTSIZE - 1)

        if dict_parameters_for_the_ax['xint']:
            if xx is None:
                raise Exception("xx has not been given.")
            x_int = range(math.ceil(min(xx)) - 1, math.ceil(
                axis(
                    xx)) + 1)  # I need to use ceil on both if min and self.axs[nb_ax] are not integers ( like 0 to 1 )
            axis.set_xticks(x_int)
        if dict_parameters_for_the_ax['yint']:
            if yy is None:
                raise Exception("yy has not been given.")
            y_int = range(min(yy), math.ceil(axis(yy)) + 1)
            axis.set_yticks(y_int)

        if dict_parameters_for_the_ax['xlim']:
            axis.set_xlim(*dict_parameters_for_the_ax['xlim'])
        if dict_parameters_for_the_ax['ylim']:
            axis.set_xlim(*dict_parameters_for_the_ax['ylim'])


        # I keep the condition. If not true, then no need to move the plot up.
        if dict_parameters_for_the_ax['parameters'] is not None \
                and dict_parameters_for_the_ax['name_parameters'] is not None:
            parameters = dict_parameters_for_the_ax['parameters']
            name_parameters = dict_parameters_for_the_ax['name_parameters']
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

            bottom, top = axis.get_ylim()
            left, right = axis.get_xlim()
            axis.text(left + (right - left) * 0.15, bottom - (top - bottom) * 0.42, sous_text,
                                  fontsize=APlot.FONTSIZE - 1)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace=0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots
        return

    def show_legend(self, nb_ax=None):
        """
        Semantics:
            Shows the legend on the chosen axes.

        Args:
            nb_ax: integer. Number of the axs we are referring to.
            If none, all the axes are looped over.

        Returns:
            Void
        """
        if nb_ax is None:
            for nb_ax_0 in range(self._nb_of_axs):
                self._axs[nb_ax_0].legend(loc='best', fontsize=APlot.FONTSIZE - 3)
        else:
            self._axs[nb_ax].legend(loc='best', fontsize=APlot.FONTSIZE - 3)
        return

    @staticmethod
    def show_plot():
        """
        SEMANTICS :
            adapter for the show pyplot function.

        Returns:
            void.

        """
        plt.show()
        return

    @staticmethod
    def save_plot(name_save_file='image'):
        """
        SEMANTICS:
            saves the plot drawn.

        Args:
            name_save_file: name of the file. Can be used to chose the path.

        Returns:
            nothing.
        """
        plt.savefig(name_save_file + '.png', dpi=800)
        return

    # section ######################################################################
    #  #############################################################################
    #  #############################################################################
    #  #############################################################################
    # PLOT FUNCTIONS

    def uni_plot(self, nb_ax, xx, yy, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_ax=None):
        """
        SEMANTICS:
            plot a single plot upon an axis.

        Args:
            nb_ax: axes on which the plot is drawn.
            xx: data xx for plot.
            yy: data yy for plot.
            dict_plot_param: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.
            dict_ax: dictionary for the parameters to customise on the axis.

        Returns:

        """
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)

        if dict_ax is not None:
            self.set_dict_ax(nb_ax = nb_ax, dict_ax = dict_ax, xx= xx, yy= yy, bis = False)
        return

    def bi_plot(self, nb_ax1, nb_ax2, xx1, yy1, xx2, yy2, dict_plot_param_1=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                dict_plot_param_2=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_ax_1=None, dict_ax_2=None):
        """
        SEMANTICS :
            Creates two plot at once.

        Args:
            nb_ax1:
            nb_ax2:
            xx1:
            yy1:
            xx2:
            yy2:
            dict_plot_param_1: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.
            dict_plot_param_2: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.
            dict_ax_1: dictionary for the parameters to customise on the axis.
            dict_ax_2: dictionary for the parameters to customise on the axis.

        Returns:

        DEPENDENCIES :
            uni_plot

        """
        self.uni_plot(nb_ax1, xx1, yy1, dict_plot_param=dict_plot_param_1, dict_ax=dict_ax_1)
        self.uni_plot(nb_ax2, xx2, yy2, dict_plot_param=dict_plot_param_2, dict_ax=dict_ax_2)
        return

    def uni_plot_ax_bis(self, nb_ax, xx, yy,
                        dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(), dict_ax=None):
        """

        Args:
            nb_ax:
            xx:
            yy:
            dict_plot_param: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.
            dict_ax: dictionary for the parameters to customise on the axis.

        Returns:

        """
        self.__plot_on_bis_ax(nb_ax, xx, yy, dict_plot_param)

        if dict_ax is not None:
            self.set_dict_ax(nb_ax = nb_ax, dict_ax = dict_ax, xx= xx, yy= yy, bis = True)
        return

    def tight_layout(self):
        self._fig.tight_layout()
        return

    def cumulative_plot(self, xx, yy, nb_ax=0):
        """
        SEMANTICS:
            Draw the cumulative distribution of the data yy, with points for the line laying at xx.

        Args:
            xx: xx where points should appear on the graph.
            yy: the data for the cumulative distribution. Real numbers.
            nb_ax: on which axis the plot is drawn.

        Returns:

        """
        dict_plot_param = {'color': 'darkorange', 'marker': 'o', 'linestyle': '-', 'markersize': 1,
                           'label': "Cumulative ratio"}
        dict_ax = {'ylabel' : 'cumulative ratio', 'ylim' : [0,1.1]}
        cum_yy = np.cumsum(np.abs(yy)) / (        np.cumsum(np.abs(yy))[-1])
        # the abs function is for making sure that it works even for negative values.
        self.__plot_on_bis_ax(nb_ax, xx, cum_yy, dict_plot_param=dict_plot_param, dict_ax=dict_ax)
        self.show_legend(nb_ax=nb_ax)
        return

    def __plot_on_bis_ax(self, nb_ax, xx, yy, dict_plot_param, dict_ax):
        """
        Semantics:
            Hidden method for plotting on a bis_axis. It first check if the axis exists, and then plots upon it.
        """
        if self._axs_bis[nb_ax] is None:  # case axis not created yet.
            self._axs_bis[nb_ax] = self._axs[nb_ax].twinx()  # instantiate a second axes that shares the same x-axis
        self.__my_plotter(nb_ax, xx, yy, dict_plot_param, bis=True)

        if dict_ax is not None:
            self.set_dict_ax(nb_ax=nb_ax, dict_ax=dict_ax, xx=xx, yy=yy, bis=True)
        return


    def hist(self, data, nb_of_ax=0, dict_param_hist=DEFAULT_DICT_HIST_PARAMETERS.copy(), dict_ax=None):
        """
        SEMANTICS : plotting histograms
        Args:
            data:
            nb_of_ax:
            dict_param_hist: dictionary with the parameters used for the plot of the histogram. examples of dict_plot_param in __my_plotter.
            dict_ax: dictionary for the parameters to customise on the axis.

        Returns:

        """
        if dict_ax is not None:
            self.set_dict_ax(nb_ax= nb_of_ax, dict_ax = dict_ax, bis = False)
        self._axs[nb_of_ax].set_xlabel("Realisation")
        self._axs[nb_of_ax].set_ylabel("Nb of realisation inside a bin.")

        function_dict.up(dict_param_hist, APlot.DEFAULT_DICT_HIST_PARAMETERS)

        try:
            # if doesn't pop, it will be catch by except.
            # that way we delete from dict_param_hist the parameter cumulative so we can give it to the plot.
            if dict_param_hist.pop("cumulative"):
                values, base, _ = self._axs[nb_of_ax].hist(data, **dict_param_hist)
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
            values, base, _ = self._axs[nb_of_ax].hist(data, **dict_param_hist)
        return

    def plot_vertical_line(self, x, yy, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """
        Semantics:
            plots a vertical line at x-axis value x, and one can chose at which points yy. 2 points are enough.
        Args:
            x:
            yy:
            nb_ax:
            dict_plot_param: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.

        Returns:

        """
        return self.uni_plot(nb_ax=nb_ax, xx=np.full(len(yy), x), yy=yy, dict_plot_param=dict_plot_param)

    def plot_function(self, function, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy(),
                      is_function_vectorised=True):
        """

        Args:
            function: callable that we want to plot.
            xx: input for function.
            nb_ax:
            dict_plot_param: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.
            is_function_vectorised: boolean, is the function passed a vectorized function that accepts function(xx)

        Returns:

        """
        # ax is an int, not necessary for uni dim case.
        if is_function_vectorised:
            xx = np.array(xx)
        yy = function(xx)

        self.__my_plotter(nb_ax, xx, yy, dict_plot_param)
        return

    def plot_line(self, a, b, xx, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """
        SEMANTICS:
            Plot a line on the chosen ax.

        Args:
            a: slope of line
            b: origin of line
            xx: data, where to have the points of the line
            nb_ax: which ax to use, should be an integer.
            dict_plot_param:  dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.

        Returns:

        """
        if function_iterable.is_a_container(a) or function_iterable.is_a_container(b):  # are a and b scalars?
            raise Error_not_allowed_input("a and b should be scalars, but containers were given.", a, b)

        function = lambda x: a * x + b
        return self.plot_function(function, xx, nb_ax=nb_ax, dict_plot_param=dict_plot_param)

    def plot_point(self, x, y, nb_ax=0, dict_plot_param=DEFAULT_DICT_PLOT_PARAMETERS.copy()):
        """
        Semantics:
            plots a single point at coordinates (x,y).

        Args:
            x:
            y:
            nb_ax:
            dict_plot_param: dictionary with the parameters used for the plot of the curve. examples of dict_plot_param in __my_plotter.

        Returns:

        Dependencies:
            uses plot_line.

        """
        return self.plot_line(a=0, b=y, xx=x, nb_ax=nb_ax, dict_plot_param=dict_plot_param)

    # section ######################################################################
    #  #############################################################################
    #  helpers for function to give indication on how to use the class.

    @staticmethod
    def help_dict_plot():
        """
        Semantics:
            print possibilities for dict_plot.
        """
        text = "dict_plot_param can be any dict with these keys (non-exhaustive): \n" \
               "    -label : string \n" \
               "    -color : 'b','g','r','c','m','y','k','w' \n" \
               "    -marker : '.',',','o','v','^','<','>','+','x' \n" \
               "    -markersize : float \n" \
               "    -linestyle : '-','--','-.',':' \n" \
               "    -linewidth : float \n" \
               "complete list at https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot \n"
        print(text)
        return

    @staticmethod
    def help_dict_ax():
        """
        Semantics:
            print possibilities for dict_ax and the default behavior.

        Dependencies:
            uses APlot_plot_dicts_for_each_axs.help_dict_ax()
        """
        APlot_plot_dicts_for_each_axs.help_dict_ax()
        return
