from unittest import TestCase

import numpy as np
from priv_lib_util.tools import function_iterable


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
