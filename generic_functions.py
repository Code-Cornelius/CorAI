# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import scipy.integrate  # for quad
import cmath  # complex numbers
import time  # computational time
import bisect


# function from overflow to integrate complex functions.
# https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return func(x).real

    def imag_func(x):
        return func(x).imag

    real_integral = scipy.integrate.quad(real_func, a, b, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])


# useful function that I put at the end of certain functions to know how long they runned.
# Print the time in a human format.
def time_computational(A, B, title="no title"):
    seconds = B - A
    seconds = round(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    beg = " Program : " + title + ", took roughly :"
    if s == 0:
        ts = ""
    if s == 1:
        ts = "{:d} second ".format(s)
    if s != 1 and s != 0:
        ts = "{:d} seconds ".format(s)

    if m == 0:
        tm = ""
    if m == 1:
        tm = "{:d} minut ".format(m)
    if m != 1 and m != 0:
        tm = "{:d} minuts ".format(m)

    if h == 0:
        th = ""
    if h == 1:
        th = "{:d} hour ".format(h)
    if h != 1 and h != 0:
        th = "{:d} hours ".format(h)

    if h == s and s == m and m == 0:
        ts = " 0.1 second "
    print(100 * '~')
    print(beg + th + tm + ts + 'to run.')
    return


# return the image of func(tt)
def evaluate_function(func, tt, *args):
    ## args if addit parameters are required
    im = np.zeros(len(tt))
    for i in range(len(tt)):
        im[i] = func(tt[i], *args)
    return im


# compute integral when the values are given.
def trapeze_int(t, y):
    ans = 0
    # corresponds to the case where the path is degenerated, only one point.
    if len(t) <= 1:
        return 0
    # corresponds to the case where the vector is constant.
    # Then in order to reduce the computation, I return the product of the length time with the constant.
    if (len(set(y)) <= 1):
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


# e is the error.
# tol is the each step tolerance
def newtons_method(f, df, x0, e=10 ** (-10), tol=10 ** (-10)):
    ## f is the function
    ## df its derivative
    ## x0    first guess
    ## e the tolerance
    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 1
    while f(x0) > e or step > tol:
        if number_of_step_crash > np.power(10, 9):
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1
        old_x0 = x0
        x0 = x0 - f(x0) / df(x0)
        step = abs(x0 - old_x0)
    return x0


# e is the error.
# tol is the each step tolerance
def newtons_method_multi(df, ddf, x0, e=10 ** (-10), tol=10 ** (-10)):
    ## df is the derivative
    ## ddf its Hessian
    ## x0    first guess
    ## e the tolerance
    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 1
    while np.linalg.norm(df(x0), 2) > e or step > tol: #I use norm 2 as criterea
        if number_of_step_crash > np.power(10, 9):
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1
        old_x0 = x0

        A = ddf(x0)
        A = np.linalg.inv(A)
        B = df(x0)

        x0 = x0 - np.matmul(A, B)
        step = abs(x0 - old_x0)
    return x0


def my_list_argmin(list):
    return list.index(min(list))

#when applied to an empty array, returns 0, which is the behaviour one would expect.
def find_smallest_rank_leq_to_K(list, K, sorted=True):
    if np.isscalar(list):
        raise Exception("Object is not a list.")
    # to generalize the function to multi dimensional arrays, I need to first know its number of dimension :
    DIM = list.ndim
    if DIM > 2:
        raise Exception("The list has too many dimensions.")
    if DIM == 1:
        # sorted argument for cases where the list is not sorted. Sorting the list is still algorithmitcaly more efficient.
        if not sorted:
            list.sort()
        return bisect.bisect_right(list, K)
    if DIM == 2:
        # I sort every line, and i search the minimal column for each row such that it satisfies certain properties.
        if not sorted:
            for i in range(np.shape(list)[0]):
                list[i, :].sort()
        # Here I had a problem, np.zeros gives back an array with floats in it. So I specify the dtype.
        ans = np.zeros(np.shape(list)[0], dtype=int)
        for i in range(np.shape(list)[0]):
            ans[i] = bisect.bisect_right(list[i, :], K)
        return ans

def test_print(str):
    print("! test ! ", str)

# do a cycle over a list
def rotate(l, n):
    return l[-n:] + l[:-n]

# using numpy
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def inverse_mult(x, vect):
    return (x * np.reciprocal(vect.astype(float)))
