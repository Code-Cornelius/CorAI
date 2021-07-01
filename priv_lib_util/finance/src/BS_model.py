# normal libraries
import math

import numpy as np
from scipy.stats import norm

# my libraries
from priv_lib_util.tools.src import function_recurrent


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
    v_sqrt = SIGMA * np.sqrt(T)
    d1 = (np.log(F / X) + (v_sqrt * v_sqrt / 2.)) / v_sqrt
    return F * function_recurrent.phi(d1) * np.sqrt(T) / DF


def BlackScholesCore(CallPutFlag, DF, F, K, T, sigma):
    """ Black Scholes Function

    One shouldn't use that function, prefer BS

    Args:
        T: Maturity
        CallPutFlag:
        DF: discount factor
        F:  Forward F c'est S_0
        K:  strike
        sigma:

    Returns:

    """
    v_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + (v_sqrt * v_sqrt / 2.)) / v_sqrt
    d2 = d1 - v_sqrt
    if CallPutFlag:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def BlackScholes(CallPutFlag, S, K, T, R, d, *, sigma=None, total_iv=None):
    """Black-Scholes Pricing Function. It is vectorised.
    Args:
        CallPutFlag (bool):
        S:  = S_0
        K:  the strike price K
        T:  maturity
        R:  continuous interest rate. Then, the total discount factor: np.exp(-R * T)
        d: dividend
        sigma: square root of the volatility (sigma, not variance). keyword argument only
        total_iv: volatility, sigma^2. keyword argument only.


    Returns:


    Precondition:
        Only sigma or TIV given, not both. Then, if TIV, T is taken as one in BS.

    """
    assert S > 0, "Price must be positive."
    if not np.ndim(K):  # if np.isscalar(K)
        assert K > 0, "Strike Price must be positive."
    else:  # ndim > 0
        assert np.all(K > 0), "Strike Price must be positive."

    assert (sigma is None or total_iv is None), "Only sigma or TIV given, not both. Both given."
    assert (sigma is not None or total_iv is not None), "Only sigma or TIV given, not both. Both none."
    if sigma is None:
        T = 1
        sigma = math.sqrt(total_iv)
    return BlackScholesCore(CallPutFlag, np.exp(-R * T), np.exp((R - d) * T) * S, K, T, sigma)
