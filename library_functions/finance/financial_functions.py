# normal libraries
import warnings

import library_functions.tools.classical_functions_optimization
import numpy as np  # maths library and arrays
import scipy.stats  # functions of statistics
# my libraries
from library_functions.tools import recurrent_functions


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
    return F * recurrent_functions.phi(d1) * np.sqrt(T) / DF


def BlackScholesCore(CallPutFlag, DF, F, X, T, SIGMA):
    """ Black Sholes Function

    One shouldn't use that function, prefer BS

    Args:
        T:
        CallPutFlag:
        DF: discount factor
        F:  Forward F c'est S_0
        X:  strike
        SIGMA:

    Returns:

    """
    vsqrt = SIGMA * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    d2 = d1 - vsqrt
    if CallPutFlag:
        return DF * (F * scipy.stats.norm.cdf(d1) - X * scipy.stats.norm.cdf(d2))
    else:
        return DF * (X * scipy.stats.norm.cdf(-d2) - F * scipy.stats.norm.cdf(-d1))


def BlackScholes(CallPutFlag, S, k, T, R, d, SIGMA):
    """Black-Scholes Pricing Function

    Args:
        CallPutFlag:
        S:  = S_0
        k:  the log strik price k, we change it for the normal strike price K = exp(k)
        T:  maturity
        R:  continuous interest rate
        d: dividend
        SIGMA:

    Returns:

    """
    K = np.exp(k)
    return BlackScholesCore(CallPutFlag, np.exp(-R * T), np.exp((R - d) * T) * S, K, T, SIGMA)


def implied_volatility_bisect(CallPutFlag, s0, k, T, R, d, experimented_price):
    """

    Args:
        CallPutFlag:
        s0: starting point of the S's,
        k:
        T:
        R:
        d:
        experimented_price:

    Returns:

    """

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        k, s0, T, r, price = args
        return price - BlackScholes(CallPutFlag, s0, k, T, r, d, vol)

    vMin, vMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try:
        return scipy.optimize.bisect(smileMin, vMin, vMax, args=(k, s0, T, R, experimented_price), xtol=1e-20,
                                     rtol=1e-15,
                                     full_output=False, disp=True)
    except:
        warnings.warn("Bisect didn't find the $\sigma_{IMP}$, returned 0.")
        return 0


def implied_volatility_newton(CallPutFlag, s0, k, T, R, d, experimented_price):
    """
    Compute Implied Volatility by newton's method.

    Args:
        CallPutFlag:
        d: dividends
        k: log strike price
        s0: initial price
        T:  maturity
        R: rate of interest rates
        experimented_price: price of the underlying

    Returns: the Implied Volatility

    """
    fx = lambda varSIGMA: BlackScholes(CallPutFlag, s0, k, T, R, d, varSIGMA) - experimented_price
    # invariant of call or put
    K = np.exp(k)
    dfx = lambda varSIGMA: BlackScholesVegaCore(np.exp(-R * T), np.exp((R - 0) * T) * s0, K, T, varSIGMA)
    try:
        return library_functions.tools.classical_functions_optimization.newtons_method(fx, dfx, 0.2)
    except:
        warnings.warn("Bisect didn't find the $\\sigma_{IMP}$, returned 0.")
        return 0
