# normal libraries

import numpy as np  # maths library and arrays
from scipy.integrate import simps


# priv_libraries


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_integral(nodes_integral, probability_weighting_density):
    # computes the expectation, of a RV taking values at nodes
    # and with each node with probability density.
    yy = nodes_integral * probability_weighting_density
    return simps(yy, nodes_integral)


def compute_price(xx_values, log_prices, density):
    assert len(density) == len(xx_values), "we want to have xx_values and density corresponding."

    strike_prices = log_prices
    ss_values = xx_values

    yy = np.repeat(ss_values[None, :], len(strike_prices), axis=0) - \
         np.repeat(strike_prices[:, None], len(ss_values), axis=1)
    C_k = simps(np.where(yy > 0, yy, 0) * density, xx_values)
    return C_k
