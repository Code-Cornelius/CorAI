# normal libraries
import warnings

import numpy as np  # maths library and arrays
import scipy.stats  # functions of statistics
from scipy.integrate import simps

# my libraries
import priv_lib_util.calculus.src.optimization
from priv_lib_util.tools import function_recurrent


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_integral(nodes_integral, density_evaluated, push=None):
    # with push = None, one computes the expectation, of a RV taking values at nodes and with each node with probability density.
    if push is None:
        integrand_evaluated = nodes_integral
    else:
        integrand_evaluated = push(nodes_integral)
    yy = integrand_evaluated * density_evaluated
    return simps(yy, nodes_integral)


def compute_price(xx_values, log_prices, density, push=None):
    assert len(density) == len(xx_values), "we want to have xx_values and density corresponding."

    if push is None:
        strike_prices = log_prices
        ss_values = xx_values
    else:
        strike_prices = push(log_prices)
        ss_values = push(xx_values)

    yy = np.repeat(ss_values[None, :], len(strike_prices), axis=0) - \
         np.repeat(strike_prices[:, None], len(ss_values), axis=1)
    C_k = simps(np.where(yy > 0, yy, 0) * density, xx_values)
    return C_k
