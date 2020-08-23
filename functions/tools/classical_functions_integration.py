import warnings

import numpy as np
import scipy.integrate



def complex_quadrature(func, a, b, *args, **kwargs):
    """
    Complex quadrature over real line, is harder than real, so here is a built function

    function from overflow to integrate complex functions.
    https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers

    Args:
        func: integrands
        a: lower bound
        b: higher bounds
        *args: for the function
        **kwargs: for the function

    Returns:
        a 3-tuple : (integral, real error, imagi error).
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
    # return the image of func(tt)
    ## args if additionnal parameters are required
    im = np.zeros(len(tt))
    for i in range(len(tt)):
        im[i] = func(tt[i], *args, **kwargs)
    return im


def trapeze_int(t, y):
    # compute integral when the values are given.

    ans = 0
    # corresponds to the case where the path is degenerated, only one point.
    if len(t) <= 1:
        warnings.warn("Object of length 1.")
        return 0


    # corresponds to the case where the vector is constant.
    # Then in order to reduce the computation, I return the product of the length time with the constant.
    if len(set(y)) <= 1:
        # this tests if there is only one value in the vector
        # I return the difference between the timr beg end
        # (because If t[0] \neq 0 there would be a problem).
        # times the first value, which is equal to all of the other by the precedant test.
        return (t[-1] - t[0]) * y[0]

    DELTA = t[1] - t[0]
    # go through all values except first and last
    for i in range(1, len(t) - 1):
        ans += DELTA * y[i]
    ans += DELTA / 2 * (y[0] + y[-1])
    return ans