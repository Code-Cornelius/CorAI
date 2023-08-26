from unittest import TestCase

import numpy as np
from corai_util.tools import function_iterable
from corai_util.tools.src.function_iterable import is_np_arr_constant


class Test_function_iterable(TestCase):
    def test_my_list_argmin(self):
        pass

    def test_find_smallest_rank_leq_to_k(self):
        pass

    def test_roundrobin(self):
        pass

    def test_mean_list(self):
        empty_list = []
        empty_array = np.array([])

        print(function_iterable.mean_list(empty_array))

    def test_is_invertible(self):
        pass

    def test_inverse_mult(self):
        pass

    def test_rotate(self):
        pass

    def test_is_iterable(self):
        pass

    def test_is_a_container(self):
        pass

    def test_replace_nans(self):
        pass

    def test_are_at_least_one_None(self):
        pass

    def test_raise_if_not_all_None(self):
        pass

    def test_is_np_arr_constant(self):
        array_positive_numbers = np.array([1, 2, 3])
        assert not is_np_arr_constant(array_positive_numbers, 0.1)
        assert not is_np_arr_constant(array_positive_numbers, 0.5)
        assert is_np_arr_constant(array_positive_numbers, 5.)

        array_constant = np.array([1, 1, 1])
        assert is_np_arr_constant(array_constant, 0.1)
        assert is_np_arr_constant(array_constant, 0.5)
        assert is_np_arr_constant(array_constant, 5.)

        array_neg_numbers = np.array([-1, -2, -3])
        assert not is_np_arr_constant(array_neg_numbers, 0.1)
        assert not is_np_arr_constant(array_neg_numbers, 0.5)
        assert is_np_arr_constant(array_neg_numbers, 5.)

        array_mixed_numbers = np.array([-1, 1, -2, 2])
        assert not is_np_arr_constant(array_mixed_numbers, 0.)
        assert not is_np_arr_constant(array_mixed_numbers, 0.5)
        assert not is_np_arr_constant(array_mixed_numbers, 2.)
