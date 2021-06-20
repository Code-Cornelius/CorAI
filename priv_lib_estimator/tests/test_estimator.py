import unittest
from unittest import TestCase
from priv_lib_estimator import Estimator

import numpy as np
import pandas as pd


class Test_Estimator(TestCase):
    def setUp(self) -> None:
        data = np.array([[1600, 4, 1],
                         [2500, 1, 2],
                         [900, 1, 1],
                         [10000, 49, 10],
                         [2500, 1, 1]])
        self.estimator = Estimator(pd.DataFrame(data=data, columns=['Size', 'People', 'number_rooms']))

    def test_from_path(self):
        pass

    def test_append(self):
        pass

    def test_apply_function_upon_data(self):
        fcts = [np.sqrt, np.square]
        separators = [['Size'], ['Size', 'People']]
        solution1 = pd.DataFrame(data=[40, 50, 30, 100, 50], columns=['Size'])
        solution2 = pd.DataFrame(data=np.array([[40, 2],
                                                [50, 1],
                                                [30, 1],
                                                [100, 7],
                                                [50, 1]]),
                                 columns=['Size', 'People'])
        solution3 = pd.DataFrame(data=[1600 * 1600, 2500 * 2500, 900 * 900, 10000 * 10000, 2500 * 2500],
                                 columns=['Size'])
        solution4 = pd.DataFrame(data=np.array([[1600 * 1600, 16],
                                                [2500 * 2500, 1],
                                                [900 * 900, 1],
                                                [10000 * 10000, 49 * 49],
                                                [2500 * 2500, 1]]),
                                 columns=['Size', 'People'])

        solution = [solution1, solution2, solution3, solution4]
        for i, fct in enumerate(fcts):
            for j, separator in enumerate(separators):
                with self.subTest(fct=fct, separator=separator):
                    ans = self.estimator.apply_function_upon_data(separators=separator, fct=fct)
                    assert (ans == solution[i * len(fcts) + j]).to_numpy().all()

    def test_apply_function_upon_data_store_it(self):
        pass

    # todo loop into estimator group mean
    @unittest.skip
    def test_estimator_mean(self):
        keys_grouping = [None, ['People'], ['People', 'number_rooms']]
        solution = [[3500], [1966 + 2 / 3, 1600, 10000], [1700, 2500, 1600, 10000]]
        for i, key in enumerate(keys_grouping):
            with self.subTest(key=key):
                if key is not None:
                    self.skipTest("Check why the test fails otherwise")
                ans = self.estimator.estimation_group_mean(columns_for_computation=['Size'],
                                                           keys_grouping=key)
                size_reshape = len(solution[i])
                assert (solution[i] == ans.to_numpy().reshape((1,
                                                               size_reshape))).all()  # converting ans to numpy (from DF) then reshaping it so comparable to list and finally compare.

        keys_grouping = [None, ['People']]
        solution = [
            [3500, 3],
            [[1966 + 2 / 3, 4 / 3],
             [1600, 1],
             [10000, 10]]
        ]

        for i, key in enumerate(keys_grouping):
            with self.subTest(key=key):
                if key is not None:
                    self.skipTest("Check why the test fails otherwise")
                ans = self.estimator.estimation_group_mean(columns_for_computation=['Size', 'number_rooms'],
                                                           keys_grouping=key)
                assert (solution[i] == ans.to_numpy()).all()

    def test_estimator_variance(self):
        pass

    def test_to_csv(self):
        pass

    def test_groupby_df(self):
        pass

    def test_df(self):
        pass
