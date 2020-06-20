# normal libraries
import numpy as np #maths library and arrays
import statistics as stat
import pandas as pd #dataframes
import seaborn as sns #envrionement for plots
from matplotlib import pyplot as plt #ploting
import scipy.stats #functions of statistics
import scipy.integrate  # for the method quad allows integration
import scipy.optimize  # for knowing when a function crosses 0, for implied volatility computation.
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time #allows to time event
import warnings
import math #quick math functions
import cmath  #complex functions

# my libraries


# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import scipy.stats  # that's for cdf in BS
import scipy.integrate  # for the method quad allows integration
import scipy.optimize  # for knowing when a function crosses 0, for implied volatility computation.
import cmath  # complex numbers
import time  # computational time
import classical_functions

def function_vanilla_option_output(x, k):
    return max(0, x - np.exp(k))

# given the parameter, and
def compute_MSE(param, true_parameter):
    return (param - true_parameter) ** 2

def phi(x):
    """
    Gaussian density PDF
    Args:
        x: optimized for np.arrays

    Returns: returns an array with the gaussian density

    """
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)



