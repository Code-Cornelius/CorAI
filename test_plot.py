# normal libraries
import numpy as np  #maths library and arrays
import statistics as stat
import pandas as pd  #dataframes
import seaborn as sns  #envrionement for plots
from matplotlib import pyplot as plt  #ploting 
import scipy.stats  #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time  #allows to time event
import warnings
import math  #quick math functions
import cmath  #complex functions

# my libraries
import classical_functions
import decorators_functions
import financial_functions
import functions_networkx
import plot_functions
import recurrent_functions
import classes.class_estimator
import classes.class_graph_estimator

np.random.seed(124)

# errors:
import errors.Error_convergence
import errors.Warning_deprecated
import errors.Error_forbidden

# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# import numpy as np
# xx = np.linspace(0,1000,10000)
# yy = xx*2
# aplot = plot_functions.APlot(how = (1,1))
# aplot.uni_plot(nb_ax = 0, xx = xx, yy = yy)
# my_list = aplot.print_register()
#
# for i in my_list:
#     i.plot_vertical_line(200, np.linspace(0,10000,100000), nb_ax=0 )
# plt.show()


# xx = np.random.random(1000)
# aplot = plot_functions.APlot(how = (1,3))
# aplot.hist(xx, 0,
#            dict_param_hist= {"bins" : 60} )
# aplot.hist(xx, 1)
# aplot.hist(xx, 2)
# plt.show()
