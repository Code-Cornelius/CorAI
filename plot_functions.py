# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import scipy.integrate  # for quad
import cmath  # complex numbers
import time  # computational time





# plot graph can plot up to 2 graphs on the same figure.
# every argument has to be a list in order to make it work.
# title and labels has to be list, where one has :
## [title 1, title 2] ; [x1label y1label, x2label y2label]
# the set of parameters is the same for the two subplots.

### don't forget to write down #plt.show() at the end !
def plot_graph(data_x, data_y, title = ["No title", "No title"], labels = ["No label","No label","No label","No label"],
               logy=[False, False], xint=[False, False], yint=[False, False],
               cum=[False, False], scater=[False, False],
               data2_x=None, data2_y=None,
               parameters=None, name_parameters=None,
               name_save_file=None):
    plt.figure(figsize=(10, 5))
    plt.grid(True)

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
        plt.xlabel(labels[0], fontsize=10)
        plt.ylabel(labels[1], fontsize=10)
        plt.title(title[0], fontsize= 8)
        x = data_x
        y = data_y
        if scater[0]:
            plt.plot(x, y, 'mo', markersize= markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', markersize=markersize, linewidth = 0.5, label=labels[1])
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
        plt.xlabel(labels[2], fontsize=10)
        plt.ylabel(labels[3], fontsize=10)
        plt.title(title[1], fontsize=8)
        x = data2_x
        y = data2_y
        if scater[1]:
            plt.plot(x, y, 'mo', markersize=markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', linewidth = 0.5, markersize=markersize, label=labels[1])
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
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title[0])
        x = data_x
        y = data_y
        if scater[0]:
            plt.plot(x, y, 'mo', markersize=markersize, label=labels[1])
        else:
            plt.plot(x, y, 'mo-', markersize=markersize, linewidth = 0.5, label=labels[1])
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
def hist(data, bins, title, labels, range=None, total_number_of_simulations = None):
    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    plt.ylabel("Nb of realisation inside a bin.")
    values, base, _ = plt.hist(data, bins=bins, density=False, alpha=0.5, color="green", range=range, label="Histogram")
    ax_bis = ax.twinx()
    values = np.append(values, 0)
    if total_number_of_simulations is not None:
        ax_bis.plot(base, np.cumsum(values) / total_number_of_simulations, color='darkorange', marker='o', linestyle='-',
                    markersize=1, label="Cumulative Histogram")
    else :
        ax_bis.plot(base, np.cumsum(values) / np.cumsum(values)[-1], color='darkorange', marker='o', linestyle='-',
                    markersize=1, label="Cumulative Histogram")
    plt.xlabel(labels)
    plt.ylabel("Proportion of the cumulative total.")
    plt.title(title, fontsize=16, y=1.02)
    ax_bis.legend()
    ax.legend()
    return

