import bisect
import itertools
import warnings

import numpy as np

#todo version array and list
def my_list_argmin(list):
    return list.index(min(list))

#todo version array and list
def find_smallest_rank_leq_to_K(list, K, sorted=True):
    # when applied to an empty array, returns 0, which is the behaviour one would expect.
    # this functions is for np.arrays
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


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # [k for k in roundrobin(list, list_dash)]
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))

#todo version array and list
def mean_list(my_list):
    # the default behaviour if list is empty, it returns 0.
    return float(sum(my_list)) / max(len(my_list), 1)


def is_invertible(a):
    # using numpy, test that a np array is invertible, first that the matrix is square, than that the rank is big enough.

    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def inverse_mult(x, vect):
    return (x * np.reciprocal(vect.astype(float)))



def rotate(l, n):
    # do a cycle over a list:
    # rotate(1) : [1,2,3,4] -> [4,1,2,3]
    # does not work with numpy array, and with integers bigger than length.
    # todo can use isinstance
    if type(l).__module__ == np.__name__: #checks if the type of list is numpy.array
        warnings.warn("The object given is not a list, but an array. The numpy function roll is used.")

        return np.roll(l,n)
    if abs(n) < len(l):
        return l[-n:] + l[:-n]
    else :
        warnings.warn("The rolling is too big, the original list is returned.")
        return l