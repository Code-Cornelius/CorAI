import warnings
import numpy as np
import scipy.integrate


def complex_quadrature(func, a, b, *args, **kwargs):
    """
    SEMANTICS : Complex quadrature over real line, is harder than real, so here is a built function

    REFERENCES : https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers

    Args:
        func: integrand
        a: lower bound of the integral
        b: higher bounds of the integral
        *args: arguments for the function
        **kwargs: Additional keyword arguments to pass as keywords arguments to
            `func`.
    Returns:
        a 3-tuple : complex valued integral, real error, imaginary error).
    """

    def real_func(x):
        return func(x).real

    def imag_func(x):
        return func(x).imag

    real_integral = scipy.integrate.quad(real_func, a, b, *args, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, *args, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0],
            real_integral[1:],
            imag_integral[1:])


def evaluate_function(func, tt, *args, **kwargs):
    """
    SEMANTICS : evaluate a function at given points using a simple for loop.

    Args:
        func: function to evaluate.
        tt: points at which to evaluate the function.
        *args: Additional arguments to pass as arguments to
            `func`.
        **kwargs:  Additional keyword arguments to pass as keywords arguments to
            `func`.
    Returns: evaluation of the function : func(tt)

    References : https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array/35216364
    this post shows that it does not matter what method you prefer, in the end, the best way is to use in-built function.
    Vectorise from numpy is just a trick for readability.

    """
    im = np.zeros(len(tt))

    for i in range(len(tt)):
        im[i] = func(tt[i], *args, **kwargs)
    return im


def trapeze_int(tt, yy):
    """ COULD USE from scipy.integrate import simps. Computes integral of a vector using trapezoidal rule

    PRECONDITIONS : t is an

    Args:
        tt: array with the times at which the integrand has to be computed. We need it to be a regular grid
        yy: image of the function at the points tt.

    Returns: integral given by trapezoidal rule.

    """

    ans = 0
    # corresponds to the case where the path is degenerated, only one point.
    if len(tt) <= 1:
        warnings.warn("Object of length 1.")
        return 0

    # corresponds to the case where the vector is constant.
    # Then in order to reduce the computation, I return the product of the length time with the constant.
    if len(set(yy)) <= 1:
        # this tests if the vector is composed of a unique value
        # I return the difference between the time beg end
        # (because If t[0] \neq 0 there would be a problem).
        # times the first value, which is equal to all of the other by the previous test.
        return (tt[-1] - tt[0]) * yy[0]

    DELTA = tt[1] - tt[0]
    # go through all values except first and last
    for i in range(1, len(tt) - 1):
        ans += DELTA * yy[i]
    ans += DELTA / 2 * (yy[0] + yy[-1])
    return ans

# section ######################################################################
#  #############################################################################
# TO REFLECT


# import scipy.integrate as integrate
# import scipy.special as special
# 
# result = integrate.quad(lambda x: special.jv(2.5, x), 0, 4.5)
# can be +/- inf boundaries.
# 
# vec_expint = np.vectorize(expint)
# 
# x = np.array([1, 3, 4])
# 
# y1 = f1(x)
# 
# from scipy.integrate import simps
# 
# trapz
# I1 = simps(y1, x)
