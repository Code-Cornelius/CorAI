# normal libraries
import warnings

import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm

# priv_libraries
from priv_lib_util.finance.src.bs_model import BlackScholes, BlackScholesVegaCore
from priv_lib_util.calculus.src.optimization import newtons_method_vectorised



# section ######################################################################
#  #############################################################################
# IV



def implied_vol_bisect(CallPutFlag, s0, K, T, R, d, experimented_price):
    """

    Args:
        CallPutFlag:
        s0: starting point of the S's,
        K: strike price
        T:
        R:
        d:
        experimented_price:

    Returns:

    """

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def _smileMin(vol, *args):
        K, s0, T, r, price = args
        return price - BlackScholes(CallPutFlag, s0, K, T, r, d, sigma=vol)

    vMin, vMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try:
        return bisect(_smileMin, vMin, vMax, args=(K, s0, T, R, experimented_price),
                      xtol=1e-20,
                      rtol=1e-15,
                      full_output=False, disp=True)
    except ValueError:
        warnings.warn("Bisect didn't find the implied volatility \sigma_{IMP}, returned NaN.")
        return np.NaN


def implied_volatility_newton(CallPutFlag, s0, K, T, R, d, experimented_price):
    """
    Compute Implied Volatility by newton's method.
    The function is vectorised regarding K.

    Args:
        CallPutFlag:
        d: dividends
        K:  strike price
        s0: initial price
        T:  maturity
        R: rate of interest rates
        experimented_price: price of the underlying

    Returns: the Implied Volatility \sigma_imp

    """
    assert len(K) == len(experimented_price)

    fx = lambda varSIGMA, indices: BlackScholes(CallPutFlag, s0, K[indices],
                                                T, R, d, sigma=varSIGMA[indices]) - experimented_price[indices]
    dfx = lambda varSIGMA, indices: BlackScholesVegaCore(np.exp(-R * T), np.exp((R - d) * T) * s0,
                                                         K[indices], T, varSIGMA[indices])
    try:
        x = np.full(len(experimented_price), 1.)
        newtons_method_vectorised(fx, dfx, x)
        return x
    except ValueError:
        warnings.warn("Bisect did not find the $\sigma_{IMP}$, returned NaN.")
        return np.NaN


# section ######################################################################
#  #############################################################################
# Total IV

def total_implied_vol_bisect(CallPutFlag, s0, K, R, d, experimented_price):
    sigma = implied_vol_bisect(CallPutFlag, s0, K, 1, R, d, experimented_price)
    return sigma * sigma


def total_implied_vol_newton(CallPutFlag, s0, K, R, d, experimented_price):
    sigma = implied_volatility_newton(CallPutFlag, s0, K, 1, R, d, experimented_price)
    return sigma * sigma

