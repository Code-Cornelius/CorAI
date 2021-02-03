# normal libraries
import numpy as np  # maths library and arrays
import scipy.integrate  # for the method quad allows integration
import scipy.optimize  # for knowing when a function crosses 0, for implied volatility computation.
import scipy.stats  # functions of statistics
# my libraries
from priv_lib_error import warning_deprecated


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def function_vanilla_option_output(x, k):
    return np.max(0, x - np.exp(k))


# given the parameter, and
def compute_MSE(param, true_parameter):
    return (param - true_parameter) ** 2


def phi(x):
    """
    Semantics:
        Gaussian density PDF.

    Args:
        x: optimized for np.arrays

    Returns:
        returns an array with the gaussian density

    """
    warning_deprecated.deprecated_function(reason="instead, in order to compute phi, chose in-built numpy fct.")
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)


# def phi_numpy(x):
#     """
#     Gaussian density PDF
#     Args:
#         x: optimized for np.arrays
#
#     Returns: returns an array with the gaussian density
#
#     """
#     return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)

def phi_numpy(x, mu, sigma):
    return scipy.stats.norm(mu, sigma).pdf(x)
