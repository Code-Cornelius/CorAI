# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import scipy.integrate  # for quad
import cmath  # complex numbers
import time  # computational time

import warnings


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
                          "markersize": 0.4
                          }

    def __init__(self, fig_dict=None, how=(1, 1), datax=None, datay=None, sharex=False,
                 sharey=False):  # sharex,y for sharing the same on plots.
        self.uni_dim = (how == (1, 1))
        # how should be a tuple with how I want to have axes.
        if datay is not None:
            if datax is not None:
                plt.plot(datax, datay, **default_param_dict)  # self ????
            else:
                plt.plot(range(len(datay)), datay, **default_param_dict)
        else:
            self.fig, self.axs = plt.subplots(*how, sharex=sharex, sharey=sharey)

    # always plotter first, then dict_updates (using the limits of the axis).
    def fig_dict_update(self, ax, fig_dict, xx=None, yy=None):
        # dict authorised:
        # {'title', 'xlabel', 'ylabel', 'xscale', 'xint', 'yint','parameters','name_parameters'}
        default_str = "Non-Defined."
        if fig_dict == None:
            fig_dict = {}

        if 'title' in fig_dict:
            ax.set_title(fig_dict[('title')], fontsize=10)
        else:
            ax.set_title(default_str, fontsize=10)

        if 'xlabel' in fig_dict:
            ax.set_xlabel(fig_dict[('xlabel')], fontsize=10)
        else:
            ax.set_xlabel(default_str, fontsize=10)

        if 'ylabel' in fig_dict:
            ax.set_ylabel(fig_dict[('ylabel')], fontsize=10)
        else:
            ax.set_ylabel(default_str, fontsize=10)

        if 'xscale' in fig_dict:
            ax.set_xscale(fig_dict[('xscale')])

        if 'xint' in fig_dict:
            if fig_dict[('xint')]:
                if xx is None:
                    raise ("xx has not been given.")
                x_int = range(math.ceil(min(xx)) - 1, math.ceil(
                    max(xx)) + 1)  # I need to use ceil on both if min and max are not integers ( like 0 to 1 )
                ax.set_xticks(x_int)
        if 'yint' in fig_dict:
            if fig_dict[('yint')]:
                if yy is None:
                    raise ("yy has not been given.")
                y_int = range(min(yy), math.ceil(max(yy)) + 1)
                ax.set_yticks(y_int)

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

            bottom, top = ax.get_ylim()
            left, right = ax.get_xlim()
            ax.text(left + (right - left) * 0.15, bottom - (top - bottom) * 0.42, sous_text, fontsize=10)
            plt.subplots_adjust(bottom=0.35, wspace=0.25, hspace = 0.5)  # bottom is how much low;
            # the amount of width reserved for blank space between subplots
            # the amount of height reserved for white space between subplots

    def __my_plotter(self, ax, xx, yy, param_dict):
        """
        A helper function to make a graph

        Parameters
        ----------
        ax : Axes
            The axes to draw to

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
        ax.grid(True)
        out = ax.plot(xx, yy, **param_dict)
        return out

    def uni_plot(self, xx, yy, param_dict=default_param_dict, fig_dict=None):
        """
        Method to have 1 plot.

        # BIANCA-HERE c'est quoi cette histoire, un tuple de longueur 1
        #  c'est un int donc je peux pas faire self.axs[0]
        """
        self.__my_plotter(self.axs, xx, yy, param_dict)
        self.fig.tight_layout()
        self.fig_dict_update(self.axs, fig_dict, xx, yy)

        return

    def bi_plot(self, xx1, yy1, xx2, yy2,
                param_dict_1=default_param_dict,
                param_dict_2=default_param_dict,
                fig_dict_1=None,
                fig_dict_2=None):
        self.__my_plotter(self.axs[0], xx1, yy1, param_dict_1)
        self.__my_plotter(self.axs[1], xx2, yy2, param_dict_2)
        self.fig.tight_layout()
        self.fig_dict_update(self.axs[0], fig_dict_1, xx=xx1, yy=yy1)
        self.fig_dict_update(self.axs[1], fig_dict_2, xx=xx2, yy=yy2)
        return

    def plot_function(self, function, xx, nb_ax=0, param_dict=default_param_dict):
        # ax is an int, not necessary for uni dim case.
        yy = [function(x) for x in xx]
        if self.uni_dim:
            self.__my_plotter(self.axs, xx, yy, param_dict)
        else:
            self.__my_plotter(self.axs[nb_ax], xx, yy, param_dict)
        return

    def plot_line(self, a, b, xx, ax=0, param_dict=default_param_dict):
        '''

        :param a:  slope of line
        :param b:  origin of line
        :param xx:  data, where to have the points of the line
        :param ax:  which ax to use, should be an integer.
        :param param_dict:  if I want to customize the plot.
        :return: nothing.
        '''
        function = lambda x: a * x + b
        return self.plot_function(function, xx, nb_ax=ax, param_dict=param_dict)

    def save_plot(self, name_save_file='image'):
        '''
        Method for saving the plot (figure) created.
        :param name_save_file:  what name
        :return:  nothing.
        '''
        plt.savefig(name_save_file + '.png', dpi=800)
        return

    def cumulative_plot(self, xx, yy, ax=0):
        if self.uni_dim:
            ax_bis = self.axs.twinx()
            ax_bis.plot(xx, np.cumsum(yy) / (np.cumsum(yy)[-1]), color='darkorange',
                        marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
            ax_bis.set_ylabel('cumulative ratio')
            ax_bis.set_ylim([0, 1.1])
            plt.legend(loc='best')
        else:
            ax_bis = self.axs[ax].twinx()
            ax_bis.plot(xx, np.cumsum(yy) / (np.cumsum(yy)[-1]), color='darkorange',
                        marker='o', linestyle='-', markersize=1, label="Cumulative ratio")
            ax_bis.set_ylabel('cumulative ratio')
            ax_bis.set_ylim([0, 1.1])
            plt.legend(loc='best')
        return
