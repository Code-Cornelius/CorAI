# global libraries
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# my libraries
from priv_lib_plot import APlot
from sklearn import metrics

from util_training import decorator_on_cpu_during_fct

sns.set()


def confusion_matrix_creator(Y, Y_predict_result, labels, title=""):
    # we create a raw confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y, Y_predict_result)

    # we get the sum of the lines and reshape it (we re going to use it for the percentage)
    cm_sum = np.sum(confusion_matrix, axis=1).reshape(-1, 1)

    # we get a matrix of percentage. (row proportion for every column)
    cm_percentage = confusion_matrix / cm_sum.astype(float) * 100

    # we create a raw array for the annotation that we will put on the final result
    annot = np.empty_like(confusion_matrix).astype(str)

    # getting the size of the matrix
    n_rows, n_cols = confusion_matrix.shape

    # here that part is for getting the right annotation at its place.
    for i in range(0, n_rows):
        # the idea is that we want, the percentage, then the number that fits in it,
        # and for diagonal elements, the sum of all the elements on the line.
        for j in range(0, n_cols):
            p = cm_percentage[i, j]
            c = confusion_matrix[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    # we set the frame
    fig, ax = plt.subplots()

    # using heatmap and setting some parameter for the confusion matrix.
    sns.set(font_scale=0.6)
    sns.heatmap(confusion_matrix, annot=annot, fmt='', ax=ax, linewidths=.5, cmap="coolwarm")
    sns.set(font_scale=1)

    # here this line and the next is for putting the meaning of the cases
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix {title}")  # title
    ax.xaxis.set_ticklabels(labels)  # putting the meaning of each column and row
    ax.yaxis.set_ticklabels(labels)
    ax.set_ylim(n_rows + 0.1, -0.1)  # expanding the graph... without it, the squares are cut in the middle.
    ax.set_xlim(-0.1, n_cols + 0.1)


@decorator_on_cpu_during_fct
def nn_plot_prediction_vs_true(net, plot_xx, plot_yy=None, plot_yy_noisy=None):
    aplot = APlot(how=(1, 1))
    plot_yy_pred = net.nn_predict_ans2cpu(plot_xx)

    if plot_yy_noisy is not None:
        aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_noisy,
                       dict_plot_param={"color": "black",
                                        "linestyle": "--",
                                        "linewidth": 0.3,
                                        "markersize": 0,
                                        "label": "Noisy Trained over Solution"
                                        })
    if plot_yy is not None:
        aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy,
                       dict_plot_param={"color": "orange",
                                        "linewidth": 1,
                                        "label": "Solution"
                                        })

    aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_pred,
                   dict_plot_param={"color": "c",
                                    "linewidth": 2,
                                    "label": "Predicted Data used for Training"
                                    }, dict_ax={"xlabel": "Time", "ylabel": "Estimation",
                                                "title": "Visualization of prediction and true solution"})
    aplot.show_legend()
    return


@decorator_on_cpu_during_fct
def nn_errors_compute_mean(net, train_X, train_Y, testing_X=None, testing_Y=None, device='cpu'):
    diff_train = (net.nn_predict(train_X) - train_Y)

    # Compute the scaled relative L1,L2, Linf validation error
    mean_relative_train_error_L1 = (torch.abs(torch.mean(diff_train) / torch.mean(torch.abs(train_Y)))).numpy()
    mean_relative_train_error_L2 = math.sqrt((torch.mean(diff_train * diff_train) /
                                              torch.mean(train_Y * train_Y)).numpy())
    mean_relative_train_error_Linf = (torch.max(torch.abs(diff_train)) / torch.mean(torch.abs(train_Y))).numpy()

    # Compute the scaled relative L2 generalisation error

    diff_test = (net.nn_predict(testing_X) - testing_Y)
    mean_relative_test_error_L1 = 0  # for return
    mean_relative_test_error_L2 = 0  # for return
    mean_relative_test_error_Linf = 0  # for return
    if testing_X is not None and testing_Y is not None:
        mean_relative_test_error_L1 = (torch.abs(torch.mean(diff_test) / torch.mean(torch.abs(testing_Y)))).numpy()
        mean_relative_test_error_L2 = (torch.mean(diff_test * diff_test) /
                                                 torch.mean(testing_Y * testing_Y)).numpy()
        mean_relative_test_error_Linf = (torch.max(torch.abs(diff_test)) / torch.mean(torch.abs(testing_Y))).numpy()

    return (mean_relative_train_error_L1, mean_relative_train_error_L2, mean_relative_train_error_Linf,
            mean_relative_test_error_L1, mean_relative_test_error_L2, mean_relative_test_error_Linf)
