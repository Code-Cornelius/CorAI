import bisect # for finding position in a list
import itertools # for roundrobin
import warnings
import collections # for the method is_a_container

import numpy as np

from library_errors.Error_not_allowed_input import Error_not_allowed_input


# todo version array and list
def my_list_argmin(my_list):
    return my_list.index(min(my_list))


# todo version array and list
def find_smallest_rank_leq_to_K(my_list, K, is_sorted=True):
    # when applied to an empty array, returns 0, which is the behaviour one would expect.
    # this functions is for np.arrays
    if np.isscalar(my_list):
        raise Exception("Object is not a list.")
    # to generalize the function to multi dimensional arrays, I need to first know its number of dimension :
    DIM = my_list.ndim
    if DIM > 2:
        raise Exception("The list has too many dimensions.")
    if DIM == 1:
        # sorted argument for cases where the list is not sorted. Sorting the list is still algorithmitcaly more efficient.
        if not is_sorted:
            my_list.sort()
        return bisect.bisect_right(my_list, K)
    if DIM == 2:
        # I sort every line, and i search the minimal column for each row such that it satisfies certain properties.
        if not is_sorted:
            for i in range(np.shape(my_list)[0]):
                my_list[i, :].sort()
        # Here I had a problem, np.zeros gives back an array with floats in it. So I specify the dtype.
        ans = np.zeros(np.shape(my_list)[0], dtype=int)
        for i in range(np.shape(my_list)[0]):
            ans[i] = bisect.bisect_right(my_list[i, :], K)
        return ans


def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # [k for k in roundrobin(list, list_dash)]
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for anext in nexts:
                yield anext()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


# todo version array and list
def mean_list(my_list):
    # the default behaviour if list is empty, it returns 0.
    if isinstance(my_list, list):
        return float(sum(my_list)) / max(len(my_list), 1)
    elif isinstance(my_list, np.ndarray):
        return np.mean(my_list)
    else:
        raise Error_not_allowed_input("Mean list takes either a list or a numpy array.")

def is_invertible(a):
    # using numpy, test that a np array is invertible, first that the matrix is square, than that the rank is big enough.

    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def inverse_mult(x, vect):
    return x * np.reciprocal(vect.astype(float))


def rotate(my_list, n):
    # do a cycle over a list:
    # rotate(1) : [1,2,3,4] -> [4,1,2,3]
    # does not work with numpy array, and with integers bigger than length.
    if isinstance(my_list, np.ndarray):  # checks if the type of list is numpy.array
        warnings.warn("The object given is not a list, but an array. The numpy function roll is used.")

        return np.roll(my_list, n)
    if abs(n) < len(my_list):
        return my_list[-n:] + my_list[:-n]
    else:
        warnings.warn("The rolling is too big, the original list is returned.")
        return my_list


def is_iterable(obj):
    """
    test whether an object is iterable.

    Args:
        obj:  anything to test

    Returns: TRUE OR FALSE

    """
    try:
        iter(obj)
    except:
        return False
    else:
        return True

def is_a_container(a_thing):
    """ checks if an object is a list, tuple, np.array...
    Args:
        a_thing: the object to test

    Returns:

    """
    return isinstance(a_thing, (collections.Sequence, np.ndarray))