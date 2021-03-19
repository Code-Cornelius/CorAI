# normal libraries
import warnings

import priv_lib_util.calculus.src.optimization
import numpy as np  # maths library and arrays
import scipy.stats  # functions of statistics
# my libraries

from priv_lib_util.tools import function_recurrent


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def BlackScholesVegaCore(DF, F, X, T, SIGMA):
    """


    Args:
        DF:
        F:
        X:
        T:
        SIGMA:

    Returns:

    """
    vsqrt = SIGMA * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    return F * function_recurrent.phi(d1) * np.sqrt(T) / DF


def BlackScholesCore(CallPutFlag, DF, F, K, T, SIGMA):
    """ Black Scholes Function

    One shouldn't use that function, prefer BS

    Args:
        T:
        CallPutFlag:
        DF: discount factor
        F:  Forward F c'est S_0
        K:  strike
        SIGMA:

    Returns:

    """
    v_sqrt = SIGMA * np.sqrt(T)
    d1 = (np.log(F / K) + (v_sqrt * v_sqrt / 2.)) / v_sqrt
    d2 = d1 - v_sqrt
    if CallPutFlag:
        return DF * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))
    else:
        return DF * (K * scipy.stats.norm.cdf(-d2) - F * scipy.stats.norm.cdf(-d1))


def BlackScholes(CallPutFlag, S, K, T, R, d, SIGMA):
    """Black-Scholes Pricing Function

    Args:
        CallPutFlag:
        S:  = S_0
        K:  the strike price k, exponential
        T:  maturity
        R:  continuous interest rate
        d: dividend
        SIGMA:

    Returns:

    """
    return BlackScholesCore(CallPutFlag, np.exp(-R * T), np.exp((R - d) * T) * S, K, T, SIGMA)


def implied_volatility_bisect(CallPutFlag, s0, K, T, R, d, experimented_price):
    """

    Args:
        CallPutFlag:
        s0: starting point of the S's,
        K: strike price (exponential)
        T:
        R:
        d:
        experimented_price:

    Returns:

    """

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        k, s0, T, r, price = args
        return price - BlackScholes(CallPutFlag, s0, K, T, r, d, vol)

    vMin, vMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try:
        return scipy.optimize.bisect(smileMin, vMin, vMax, args=(K, s0, T, R, experimented_price), xtol=1e-20,
                                     rtol=1e-15,
                                     full_output=False, disp=True)
    except ValueError:
        warnings.warn("Bisect didn't find the $\sigma_{IMP}$, returned 0.")
        return 0


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

    Returns: the Implied Volatility

    """
    fx = lambda varSIGMA: BlackScholes(CallPutFlag, s0, K, T, R, d, varSIGMA) - experimented_price
    dfx = lambda varSIGMA: BlackScholesVegaCore(np.exp(-R * T), np.exp((R - 0) * T) * s0, K, T, varSIGMA)
    try:
        return priv_lib_util.calculus.optimization.newtons_method(fx, dfx, 0.2)
    except ValueError:
        warnings.warn("Bisect didn't find the $\\sigma_{IMP}$, returned 0.")
        return 0
