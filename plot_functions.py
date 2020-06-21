# normal libraries
import numpy as np #maths library and arrays
import statistics as stat
import pandas as pd #dataframes
import seaborn as sns #envrionement for plots
from matplotlib import pyplot as plt #ploting
import scipy.stats #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time #allows to time event
import warnings
import math #quick math functions
import cmath  #complex functions

# my libraries

# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# plot graph can plot up to 2 graphs on the same figure.
# every argument has to be a list in order to make it work.
# title and labels has to be list, where one has :
## [title 1, title 2] ; [x1label y1label, x2label y2label]
# the set of parameters is the same for the two subplots.

### don't forget to write down #plt.show() at the end !
def plot_graph(data_x, data_y, title=["No title", "No title"], labels=["No label", "No label", "No label", "No label"],
               logy=[False, False], xint=[False, False], yint=[False, False],
               cum=[False, False], scater=[False, False],
               data2_x=None, data2_y=None,
               parameters=None, name_parameters=None,
               name_save_file=None):
    plt.figure(figsize=(10, 5))
    warnings.warn("Function is deprecated. Use new version with class APlot.")

    markersize = 0.4
    if parameters is not None:
        nb_parameters = len(parameters)
        sous_text = " Parameters : \n"
        for i in range(nb_parameters):
            sous_text += str(name_parameters[i]) + " = {}".format(parameters[i])
            # end of the list, we finish by a full stop.
            if i == nb_parameters - 1:
                sous_text += "."
            # certain chosen number of parameters by line, globally, 3 by line.
            # There shoudln't be more than 16 parameters
            elif i in [4, 7, 10, 13, 16]:
                sous_text += ", \n "
            # otherwise, just keep writing on the same line.
            else:
                sous_text += ", "
    if data2_x is not None:
        ax = plt.subplot(121)
        plt.grid(True)
        plt.xlabel(labels[0], fontsize=10)
        plt.ylabel(labels[1], fontsize=10)
        plt.title(title[0], fontsize=10)
        x = data_x
        y = data_y
        if scater[0]:
            plt.plot(x, y, 'mo', markersize=markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', markersize=markersize, linewidth=0.5, label=labels[1])
        if logy[0]:
            plt.xscale('log')
        # for cumulative, I use another axis on the right.
        if cum[0]:
            ax_bis = ax.twinx()
            ax_bis.plot(x, np.cumsum(y) / (np.cumsum(y)[-1]), color='darkorange',
                        marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
            ax_bis.set_ylabel('cumulative ratio')
            ax_bis.set_ylim([0, 1.1])
            plt.legend(loc='best')
        # change ticks for every integers
        if xint[0]:
            x_int = range(min(x), math.ceil(max(x)) + 1)
            plt.xticks(x_int)
        if yint[0]:
            y_int = range(min(y), math.ceil(max(y)) + 2)
            plt.yticks(y_int)

        if parameters is not None:
            bottom, top = plt.ylim()
            left, right = plt.xlim()
            plt.text(left + (right - left) * 0.2, bottom - (top - bottom) * 0.43, sous_text, fontsize=10)
            plt.subplots_adjust(bottom=0.3, wspace=0.35)

        ax = plt.subplot(122)
        plt.grid(True)
        plt.xlabel(labels[2], fontsize=10)
        plt.ylabel(labels[3], fontsize=10)
        plt.title(title[1], fontsize=10)
        x = data2_x
        y = data2_y
        if scater[1]:
            plt.plot(x, y, 'mo', markersize=markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', linewidth=0.5, markersize=markersize, label=labels[1])
        if logy[1]:
            plt.xscale('log')
        # for cumulative, I use another axis on the right.
        if cum[1]:
            ax_bis = ax.twinx()
            ax_bis.plot(x, np.cumsum(y) / (np.cumsum(y)[-1]), color='darkorange',
                        marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
            ax_bis.set_ylabel('cumulative ratio')
            ax_bis.set_ylim([0, 1.1])
            plt.legend(loc='best')
        # change ticks for every integers
        if xint[1]:
            x_int = range(min(x), math.ceil(max(x)) + 1)
            plt.xticks(x_int)
        if yint[1]:
            y_int = range(min(y), math.ceil(max(y)) + 2)
            plt.yticks(y_int)
        if name_save_file is not None:
            plt.savefig(name_save_file + '.png', dpi=1000)

    else:
        plt.grid(True)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title[0])
        x = data_x
        y = data_y
        if scater[0]:
            plt.plot(x, y, 'mo', markersize=markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', markersize=markersize, linewidth=0.5, label=labels[1])
        if logy[0]:
            plt.xscale('log')
        # for cumulative, I use another axis on the right.
        if cum[0]:
            ax = plt.subplot()
            ax_bis = ax.twinx()
            ax_bis.plot(x, np.cumsum(y) / (np.cumsum(y)[-1]), color='darkorange',
                        marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
            ax_bis.set_ylabel('cumulative ratio')
            ax_bis.set_ylim([0, 1.1])
            plt.legend(loc='best')
        # change ticks for every integers
        if xint[0]:
            x_int = range(min(x), math.ceil(max(x)) + 1)
            plt.xticks(x_int)
        if yint[0]:
            y_int = range(min(y), math.ceil(max(y)) + 2)
            plt.yticks(y_int)

        if parameters is not None:
            bottom, top = plt.ylim()
            left, right = plt.xlim()
            plt.text(left + (right - left) * 0.2, bottom - (top - bottom) * 0.43, sous_text, fontsize=10)
            plt.subplots_adjust(bottom=0.3)
        if name_save_file is not None:
            plt.savefig(name_save_file + '.png', dpi=1000)
    return


# function for plotting histograms
def hist(data, bins, title, labels, range=None, total_number_of_simulations=None):
    warnings.warn("Function is deprecated. Use new version.")
    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    plt.ylabel("Nb of realisation inside a bin.")
    values, base, _ = plt.hist(data, bins=bins, density=False, alpha=0.5, color="green", range=range, label="Histogram")
    ax_bis = ax.twinx()
    values = np.append(values, 0)
    if total_number_of_simulations is not None:
        ax_bis.plot(base, np.cumsum(values) / total_number_of_simulations, color='darkorange', marker='o',
                    linestyle='-',
                    markersize=1, label="Cumulative Histogram")
    else:
        ax_bis.plot(base, np.cumsum(values) / np.cumsum(values)[-1], color='darkorange', marker='o', linestyle='-',
                    markersize=1, label="Cumulative Histogram")
    plt.xlabel(labels)
    plt.ylabel("Proportion of the cumulative total.")
    plt.title(title, fontsize=16, y=1.02)
    ax_bis.legend()
    ax.legend()
    return


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
######################################### new plot functions


class APlot:
    # APlot is the class for my plots. APlot is one figure.

    default_param_dict = {"color": 'm',
                          "linestyle": "solid",
                          "linewidth": 0.5,
                          "marker": "o",
                          "markersize": 0.4,
                          "label": "plot"
                          }

    def __init__(self, figsize =(7, 5), how=(1, 1), datax=None, datay=None, sharex=False,
                 sharey=False):  # sharex,y for sharing the same on plots.
        # how should be a tuple with how I want to have axes.
        if datay is not None:
            if datax is not None:
                plt.figure(figsize= figsize)
                plt.plot(datax, datay, **self.default_param_dict)  # BIANCA-HERE self ????
            else:
                plt.figure(figsize=figsize)
                plt.plot(range(len(datay)), datay, **self.default_param_dict)
        else:
            self.fig, self.axs = plt.subplots(*how, sharex=sharex, sharey=sharey, figsize = figsize)
            self.uni_dim = (how == (1, 1))
        # two cases, if it is uni_dim, I put self.axs into a list. Otherwise, it is already a list.
        if self.uni_dim:
            self.axs = [self.axs]
        else :
            self.axs = self.axs.flatten()
        # now, self.axs is always a list.
        self.nb_of_axs = how[0] * how[1] # nb of axes upon which I can plot

    # BIANCA-HERE create decorator
    def check_axs(self, ax):
        if ax < 0 :
            # BIANCA RAISE ERROR
            raise IndexError("Index is negative.")
            pass
        if ax >= self.nb_of_axs:
            warnings.warn("Axs given is out of bounds. I plot upon the first axis.")
            ax = 0
        return ax

    # always plotter first, then dict_updates (using the limits of the axis).
    def fig_dict_update(self, nb_ax, fig_dict, xx=None, yy=None):
        # dict authorised:
        # {'title', 'xlabel', 'ylabel', 'xscale', 'xint', 'yint','parameters','name_parameters'}
        nb_ax = self.check_axs(nb_ax)
        default_str = "Non-Defined."
        if fig_dict == None:
            fig_dict = {}

        if 'title' in fig_dict:
            self.axs[nb_ax].set_title(fig_dict[('title')], fontsize=10)
        else:
            self.axs[nb_ax].set_title(default_str, fontsize=10)

        if 'xlabel' in fig_dict:
            self.axs[nb_ax].set_xlabel(fig_dict[('xlabel')], fontsize=10)
        else:
            self.axs[nb_ax].set_xlabel(default_str, fontsize=10)

        if 'ylabel' in fig_dict:
            self.axs[nb_ax].set_ylabel(fig_dict[('ylabel')], fontsize=10)
        else:
            self.axs[nb_ax].set_ylabel(default_str, fontsize=10)

        if 'xscale' in fig_dict:
            self.axs[nb_ax].set_xscale(fig_dict[('xscale')])

        if 'xint' in fig_dict:
            if fig_dict[('xint')]:
                if xx is None:
                    raise ("xx has not been given.")
                x_int = range(math.ceil(min(xx)) - 1, math.ceil(
                    self.axs[nb_ax](xx)) + 1)  # I need to use ceil on both if min and mself.axs[nb_ax] are not integers ( like 0 to 1 )
                self.axs[nb_ax].set_xticks(x_int)
        if 'yint' in fig_dict:
            if fig_dict[('yint')]:
                if yy is None:
                    raise ("yy has not been given.")
                y_int = range(min(yy), math.ceil(self.axs[nb_ax](yy)) + 1)
                self.axs[nb_ax].set_yticks(y_int)

        if 'parameters' in fig_dict and 'name_parameters' in fig_dict:
            #### check if this is correct
            # or fig ?
            parameters = fig_dict[('parameters')]
            name_parameters = fig_dict[('name_parameters')]
            nb_parameters = len(parameters)
            sous_text = " Parameters : \n"
            for i in range(nb_parameters):
                sous_text += str(name_parameters[i]) + " = {}".format(parameters[i])
                # end of the list, we finish by a full stop.
                if i == nb_parameters - 1:
                    sous_text += "."
                # certain chosen number of parameters by line, globally, 3 by line.
                # There shoudln't be more than 16 parameters
                elif i in [4, 7, 10, 13, 16]:
                    sous_text += ", \n "
                # otherwise, just keep writing on the same line.
                else:
                    sous_text += ", "

            bottom, top = self.axs[nb_ax].get_ylim()
            left, right = self.axs[nb_ax].get_xlim()
            self.axs[nb_ax].text(left + (right - left) * 0.15, bottom - (top - bottom) * 0.42, sous_text, fontsize=10)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace = 0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots

    def __my_plotter(self, nb_ax, xx, yy, param_dict):
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

        param_dict : dict
           Dictionary of kwargs to pass to ax.plot

        Returns
        -------
        out : list
            list of artists added
        """
        nb_ax = self.check_axs(nb_ax)
        self.axs[nb_ax].grid(True)
        out = self.axs[nb_ax].plot(xx, yy, **param_dict)
        return out

    def uni_plot(self, nb_ax, xx, yy, param_dict=default_param_dict, fig_dict=None):
        """
        Method to have 1 plot. Upon nb_ax (int)
        """
        self.__my_plotter(nb_ax, xx, yy, param_dict)
        self.fig.tight_layout()
        self.fig_dict_update(nb_ax, fig_dict, xx, yy)

        return

    def bi_plot(self, nb_ax1, nb_ax2, xx1, yy1, xx2, yy2,
                param_dict_1=default_param_dict,
                param_dict_2=default_param_dict,
                fig_dict_1=None,
                fig_dict_2=None):
        self.uni_plot(nb_ax1, xx1, yy1, param_dict=param_dict_1, fig_dict=fig_dict_1)
        self.uni_plot(nb_ax2, xx2, yy2, param_dict=param_dict_2, fig_dict=fig_dict_2)
        return

    def plot_function(self, function, xx, nb_ax=0, param_dict=default_param_dict):
        # ax is an int, not necessary for uni dim case.
        yy = [function(x) for x in xx]
        self.__my_plotter(nb_ax, xx, yy, param_dict)
        return

    def plot_line(self, a, b, xx, nb_ax=0, param_dict=default_param_dict):
        """
        Plot a line on the chosen ax.

        Args:
            a: slope of line
            b: origin of line
            xx: data, where to have the points of the line
            nb_ax: which ax to use, should be an integer.
            param_dict:  if I want to customize the plot.

        Returns:

        """
        function = lambda x: a * x + b
        return self.plot_function(function, xx, nb_ax=nb_ax, param_dict=param_dict)

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

    default_param_dict_hist = {'bins': 20,
                               "color": 'green', 'range': None,
                               'label': "Histogram", "cumulative": True}
    # function for plotting histograms
    def hist(self, data, nb_of_ax,
             param_dict_hist = default_param_dict_hist,
             fig_dict = None):
        self.fig_dict_update(nb_of_ax, fig_dict)
        self.axs[nb_of_ax].set_ylabel("Nb of realisation inside a bin.")
        try :
            #if doesn't pop, it will be catch by except.
            if param_dict_hist.pop("cumulative"):
                values, base, _ = self.axs[nb_of_ax].hist(data, density=False, alpha=0.5, **param_dict_hist)
                ax_bis = self.axs[nb_of_ax].twinx()
                values = np.append(values, 0)
                # I add 0 because I want to create the last line, which does not go up.
                # I put then 0 in order to have no evolution with cumsum.

                if 'total_number_of_simulations' in param_dict_hist:
                    ax_bis.plot(base, np.cumsum(values) / param_dict_hist[('total_number_of_simulations')],
                                color='darkorange', marker='o',
                                linestyle='-',
                                markersize=1, label="Cumulative Histogram")
                else:
                    ax_bis.plot(base, np.cumsum(values) / np.cumsum(values)[-1],
                                color='darkorange', marker='o', linestyle='-',
                                markersize=1, label="Cumulative Histogram")
                ax_bis.set_ylabel("Proportion of the cumulative total.")

        except KeyError: #no cumulative in the hist.
            values, base, _ = self.axs[nb_of_ax].hist(data, density=False, alpha=0.5, **param_dict_hist)
        return




    def show_legend(self, nb_ax = None):
        # as usually, nb_ax is an integer.
        # if ax is none, then every nb_ax is showing the nb_ax.
        if nb_ax is None:
            for nb_ax_0 in range(self.nb_of_axs):
                self.axs[nb_ax_0].legend(loc='best')
        else:
            self.axs[nb_ax].legend(loc='best')
        return

    def save_plot(self, name_save_file='image'):
        """
        Method for saving the plot (figure) created.

        Args:
            name_save_file: name of the file

        Returns: nothing.
        """
        plt.savefig(name_save_file + '.png', dpi=800)
        return