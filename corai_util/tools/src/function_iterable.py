import bisect  # for finding position in a list
import collections  # for the method is_a_container
import itertools  # for roundrobin
import warnings
import re

import numpy as np
from corai_error import Error_not_allowed_input
from corai_error import numpy_function_used


def my_list_argmin(my_list):
    # TODO MULTIPLE LIST, do for list of lists.
    """
    Semantics:
        Returns the indices of the minimum value.

    Returns:
        minimal value's index.

    """
    if isinstance(my_list, np.ndarray):
        numpy_function_used(which_function_used_instead="argmin")
        return np.argmax(my_list)
    return my_list.index(min(my_list))


def my_list_argmax(iterable):
    # TODO MULTIPLE LIST, do for list of lists.
    """
    Semantics:
        Returns the indices of the maximum value.
    Args:
        iterable:

    Returns:
        maximum value's index.

    """
    if isinstance(iterable, np.ndarray):
        numpy_function_used(which_function_used_instead="argmax")
        return np.argmax(iterable)
    return iterable.index(max(iterable))


def find_smallest_rank_leq_to_K(my_list, K, is_sorted=True):
    # todo version array and list

    # when applied to an empty array, returns 0, which is the behaviour one would expect.
    # this functions is for np.arrays
    if np.isscalar(my_list):
        raise Exception("Object is not a list.")
    # to generalize the function to multi dimensional arrays, I need to first know its number of dimension:
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
    """
    Semantics:
        [k for k in roundrobin(list, list_dash)]

    References:
        George Sakkis

    Examples:
        roundrobin('ABC', 'D', 'EF') --> A D E B F C

    """
    #
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for anext in nexts:
                yield anext()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def mean_list(iterable):
    # TODO MULTIPLE LIST, do for list of lists.
    """
    Semantics:
        Computes the mean of a list. Depending on the type of iterable, it uses different technics.

    Precondition:
        Iterable is iterable with only one axis.

    Returns:
        The average of the input.

    exception guaranties:
        If the iterable is empty, there is an error.

    """
    # if iterable is empty, raises a value error.
    if len(iterable) == 0:
        raise ValueError("The list is empty.")

    if isinstance(iterable, list):
        return float(sum(iterable)) / max(len(iterable), 1)

    elif isinstance(iterable, np.ndarray):
        numpy_function_used(which_function_used_instead="mean")
        return np.nanmean(iterable)

    else:
        raise Error_not_allowed_input("Mean list takes either a list or a numpy array.")


def is_numpy_matrix_invertible(a):
    """
    Semantics:
        For a numpy matrix, test that a np array is invertible:

    condition 1:
        The matrix is square,

    condition 2:
        The rank is full.
    """
    #

    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def division_numpy(x, a_vector):
    """
    Semantics:
        Computes x * 1/ a_vector where the division is taken component wise.

    Preconditions:
        x has to be multiplicable by a np.array.
        a_vector has to be convertible to float type.

    Returns: x ./ a_vector

    """
    return x * np.reciprocal(a_vector.astype(float))


def rotate(iterable, how_much_rotate):
    """
    Semantics:
        Do a cycle over an iterable. If numpy array, optimal method used.

    Returns:
        rotated iterable;

    Precondition:
        how_much_rotate < len(iterable)

    Warnings:
        if numpy function is used.
        if not how_much_rotate < len(iterable).

    Examples:
        rotate(list, 1): [1,2,3,4] -> [4,1,2,3]

    """
    if isinstance(iterable, np.ndarray):  # checks if the type of list is numpy.array
        numpy_function_used(which_function_used_instead="roll")
        return np.roll(iterable, how_much_rotate)

    if abs(how_much_rotate) < len(iterable):
        return iterable[-how_much_rotate:] + iterable[:-how_much_rotate]
    else:
        warnings.warn("The rolling is too big, the original list is returned.")
        return iterable


def is_iterable(obj):
    """
    Semantics:
        Test whether an object is iterable. By definition, it is an iterable if it has an iter method implemented.

    Returns:
        Boolean

    """
    try:
        iter(obj)
    except:
        return False
    else:
        return True


def is_a_container(a_thing):
    """
    Semantics:
        Checks if an object is a list, tuple, np.array... part of (collections.Sequence, np.ndarray))
        At the essence, we need to see if iterable and supports indexing.
    Returns:
        Boolean

    """
    return isinstance(a_thing, (collections.abc.Sequence, np.ndarray))


def replace_nans_numpy(np_array):
    """
    Semantics:
        Replace all the nans by 0 inside a numpy array.
    """
    if not isinstance(np_array, np.ndarray):  # checks if the type of list is numpy.array
        warnings.warn("Object is not a np.ndarray!")
    where_are_NaNs = np.isnan(np_array)
    np_array[where_are_NaNs] = 0
    return np_array


def are_at_least_one_None(list_parameters):
    """returns list_parameters.at least one.is_None"""
    for parameter in list_parameters:
        if parameter is None:
            return True
        else:
            continue
    return False


def raise_if_not_all_None(list_parameters):
    """ if one is not None, throws an error"""
    for parameter in list_parameters:
        if not parameter is None:
            raise ValueError("Given a parameter not None while the others are. "
                             "Is it a mistake ? Parameter not None : " + str(parameter))
    return


def is_np_arr_constant(arr, tol):
    """
    Condition if all values are equal (up to some allowed error).
    Args:
        arr:
        tol: percent error of the mean, unless the mean is zero, then it is the exact value.

    Returns:

    """
    the_mean = np.mean(arr)
    if the_mean <= 1E-8: # threshold.
        cdt1 = arr <= tol
        cdt2 = arr >= -tol
    else:
        cdt1 = np.abs(arr - the_mean) <= abs(the_mean) * tol
        cdt2 = np.abs(arr - the_mean) >= -abs(the_mean) * tol
    if np.all(cdt1 & cdt2):
        return True
    else:
        return False

# print(1)
# arr = np.array([1, 2, 3])
# print(is_np_arr_constant(arr, 0.1))
# print(is_np_arr_constant(arr, 0.5))
# print(is_np_arr_constant(arr, 5.))
#
# print(2)
# arr = np.array([-1, -2, -3])
# print(is_np_arr_constant(arr, 0.1))
# print(is_np_arr_constant(arr, 0.5))
# print(is_np_arr_constant(arr, 5.))
#
# print(3)
# arr = np.array([-1, 1, -2, 2])
# print(is_np_arr_constant(arr, 0.))
# print(is_np_arr_constant(arr, 0.5))
# print(is_np_arr_constant(arr, 2.))




def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
