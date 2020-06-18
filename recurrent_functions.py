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
    return F * phi(d1) * np.sqrt(T) / DF


#### Black Sholes Function
def BlackScholesCore(CallPutFlag, DF, F, X, T, SIGMA):
    '''
    One shouldn't use that function, prefer BS

    Args:
        CallPutFlag:
        DF: discount factor
        F:  Forward F c'est S_0
        X:  strike
        SIGMA:

    Returns:

    '''
    vsqrt = SIGMA * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    d2 = d1 - vsqrt
    if CallPutFlag:
        return DF * (F * scipy.stats.norm.cdf(d1) - X * scipy.stats.norm.cdf(d2))
    else:
        return DF * (X * scipy.stats.norm.cdf(-d2) - F * scipy.stats.norm.cdf(-d1))


##  Black-Scholes Pricing Function
def BlackScholes(CallPutFlag, S, k, T, R, d, SIGMA):
    K = np.exp(k)
    ## X: strike price, exp(k)
    ## S c'est S_0
    ## R, d: continuous interest rate and dividend
    return BlackScholesCore(CallPutFlag, np.exp(-R * T), np.exp((R - d) * T) * S, K, T, SIGMA)



def implied_volatility_bisect(k, s0, T, R, experimented_price):
    ## s0 starting point of the S's,
    ## S realisation of the S_T

    # Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        k, s0, T, r, price = args
        return price - BlackScholes(True, s0, k, T, r, 0., vol)

    vMin, vMax = 0.00001, 20.
    # in order to find the implied volatility, one has to find the value at which smileMin crosses zero.
    try :
        return scipy.optimize.bisect(smileMin, vMin, vMax, args=( k, s0, T, R, experimented_price), xtol=1e-20, rtol=1e-15,
                                 full_output=False, disp=True)
    except :
        warnings.warn("Bisect didn't find the $\sigma_{IMP}$, returned 0.")
        return 0



def implied_volatility_newton(k, s0, T, R, experimented_price):
    """
    Compute Implied Volatility by newton's method.

    Args:
        k: log strike price
        s0: initial price
        T:  maturity
        R: rate of interest rates
        experimented_price: price of the underlying

    Returns: the Implied Volatility

    """
    ## s0 starting point of the S's,
    ## S realisation of the S_T

    fx = lambda varSIGMA : BlackScholes(True, s0, k, T, R, 0, varSIGMA) - experimented_price
    # invariant of call or put
    K = np.exp(k)
    dfx = lambda varSIGMA : BlackScholesVegaCore(   np.exp(-R * T), np.exp((R - 0) * T) * s0, K, T, varSIGMA   )

    return classical_functions.newtons_method(fx, dfx, 0.2)


