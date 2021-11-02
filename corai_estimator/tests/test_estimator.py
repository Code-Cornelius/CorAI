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
        self.estimator = Estimator(df = pd.DataFrame(data=data, columns=['Size', 'People', 'number_rooms'], dtype='int32'))

    def test_from_path(self):
        pass

    def test_append(self):
        pass

    def test_apply_function_upon_data(self):
        # TODO LOOK INTO
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
        # TODO LOOK INTO
        pass


    def test_to_csv_and_from_csv(self):
        self.estimator.to_csv("test.csv")
        self.estimator2 = Estimator.from_csv("test.csv", dtype='int32') # saving might change types.
        assert self.estimator.df.equals(self.estimator2.df) # equality in terms of values


    def test_groupby_df(self):
        pass

    def test_df(self):
        pass
