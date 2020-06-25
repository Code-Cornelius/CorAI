# normal libraries
import numpy as np #maths library and arrays
import statistics as stat
import pandas as pd #dataframes
import seaborn as sns #envrionement for plots
from matplotlib import pyplot as plt #ploting
import scipy.stats #functions of statistics
import scipy.integrate  # for quad
import scipy.integrate  # for the method quad allows integration
import scipy.optimize  # for knowing when a function crosses 0, for implied volatility computation.
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time #allows to time event
import warnings
import math #quick math functions
import cmath  #complex functions
import bisect


# my libraries

# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# function from overflow to integrate complex functions.
# https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
def complex_quadrature(func, a, b, *args, **kwargs):
    '''
    Complex quadra over real line, is harder than real, so here is a built function

    Args:
        func: integrands
        a: lower bound
        b: higher bounds
        *args: for the function
        **kwargs: for the function

    Returns:
        a 3-tuple : (integral, real error, imagi error).
    '''
    def real_func(x):
        return func(x).real

    def imag_func(x):
        return func(x).imag

    real_integral = scipy.integrate.quad(real_func, a, b, *args, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, *args, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0],
            real_integral[1:],
            imag_integral[1:])


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
def evaluate_function(func, tt, *args, **kwargs):
    ## args if additionnal parameters are required
    im = np.zeros(len(tt))
    for i in range(len(tt)):
        im[i] = func(tt[i], *args, **kwargs)
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
        if step == np.inf or number_of_step_crash > np.power(10, 9): # function too flat.
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

# do a cycle over a list:
# rotate(1) : [1,2,3,4] -> [4,1,2,3]
# does not work with numpy array, and with integers bigger than length.
def rotate(l, n):
    if type(l).__module__ == np.__name__: #checks if the type of list is numpy.array
        # BIANCA-HERE can I trigger here a warning ? How?
        warnings.warn("The object given is not a list, but an array. The numpy function roll is used.")

        return np.roll(l,n)
    if abs(n) < len(l):
        return l[-n:] + l[:-n]
    else :
        # BIANCA-HERE can I trigger here a warning ? How?
        warnings.warn("The rolling is too big, the original list is returned.")
        return l


# using numpy, test that a np array is invertible, first that the matrix is square, than that the rank is big enough.
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def inverse_mult(x, vect):
    return (x * np.reciprocal(vect.astype(float)))



def tuple_to_str(my_tuple):
    my_str = ''
    for i in range(len(my_tuple)):
        my_str += str(my_tuple[i]) + '_'

    return my_str


def up(my_dict, new_dict):
    '''
    If a key from new_dict is not defined in my_dict, add it. The behaviour is almost like update.
    Args:
        my_dict: old to update
        new_dict: the new information

    Returns: nothing, the dict is updated ! no copy !

    '''
    for key in new_dict:
        if key not in my_dict:
            my_dict[key] = new_dict[key]
    return