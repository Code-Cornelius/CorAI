# normal libraries
import numpy as np
import math

from priv_lib_util.calculus.src.optimization import newtons_method
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import warnings

# my libraries
from priv_lib_util.finance.src.BS_model import BlackScholes, BlackScholesVegaCore


# section ######################################################################
#  #############################################################################
# IV


def implied_volatility_bisect(CallPutFlag, s0, K, T, R, d, experimented_price, push=None):
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
    if push is None:
        pass
    else:
        K = push(K)

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        K, s0, T, r, price = args
        return price - BlackScholes(CallPutFlag, s0, K, T, r, d, sigma=vol)

    vMin, vMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try:
        return bisect(smileMin, vMin, vMax, args=(K, s0, T, R, experimented_price),
                      xtol=1e-20,
                      rtol=1e-15,
                      full_output=False, disp=True)
    except ValueError:
        warnings.warn("Bisect didn't find the implied volatility \sigma_{IMP}, returned NaN.")
        return np.NaN


def implied_volatility_newton(CallPutFlag, s0, K, T, R, d, experimented_price):
    """
    Compute Implied Volatility by newton's method.

    Args:
        CallPutFlag:
        d: dividends
        K:  strike price (exponential)
        s0: initial price
        T:  maturity
        R: rate of interest rates
        experimented_price: price of the underlying

    Returns: the Implied Volatility \sigma_IV

    """
    assert len(K) == len(experimented_price)

    fx = lambda varSIGMA, indices: BlackScholes(CallPutFlag, s0, K[indices], T, R, d, sigma=varSIGMA[indices]) - \
                                   experimented_price[indices]
    dfx = lambda varSIGMA, indices: BlackScholesVegaCore(np.exp(-R * T), np.exp((R - d) * T) * s0,
                                                         K[indices], T, varSIGMA[indices])
    try:
        return newtons_method(fx, dfx, np.full(len(experimented_price), 1.))
    except ValueError:
        warnings.warn("Bisect didn't find the $\sigma_{IMP}$, returned NaN.")
        return np.NaN


# section ######################################################################
#  #############################################################################
# Total IV

def TIV_bisect(CallPutFlag, s0, K, R, d, experimented_price, push=None):
    """

    Args:
        CallPutFlag:
        s0: starting point of the S's,
        K: strike price
        R:
        d:
        experimented_price:

    Returns:

    """
    if push is None:
        pass
    else:
        K = push(K)

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(TIV, *args):
        K, s0, r, price = args
        return price - BlackScholes(CallPutFlag, s0, K, r, d, total_iv=TIV)

    tivMin, tivMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try:
        return bisect(smileMin, tivMin, tivMax, args=(K, s0, R, experimented_price),
                      xtol=1e-20, rtol=1e-15,
                      full_output=False, disp=True)
    except ValueError:
        warnings.warn("Bisect didn't find the total implied variance T*sigma_{IMP}^2, returned NaN.")
        return np.NaN
