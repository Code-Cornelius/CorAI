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
import errors.error_convergence as error

# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def function_vanilla_option_output(x, k):
    return np.max(0, x - np.exp(k))

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
    error.deprecated_function(reason="phi chose numpy.")
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)



def phi_numpy(x):
    """
    Gaussian density PDF
    Args:
        x: optimized for np.arrays

    Returns: returns an array with the gaussian density

    """
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)