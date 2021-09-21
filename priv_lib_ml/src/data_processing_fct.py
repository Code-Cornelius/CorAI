import os

import numpy as np
from priv_lib_util.tools.src.function_writer import list_of_dicts_to_json
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def pipeline_scaling_minimax(df):
    minimax = MinMaxScaler(feature_range=(0, 1))
    minimax.fit(df)
    return minimax, minimax.transform(df)


def pipeline_scaling_normal(df):
    standar_normalis = StandardScaler()  # norm l2 and gives a N_0,1, on each column.
    standar_normalis.fit(df.values.reshape(-1, 1))
    return standar_normalis, standar_normalis.transform(df.values.reshape(-1, 1)).reshape(-1)


def add_column_cyclical_features(df, col_name_time, period, start_num=0):
    """
    Semantics:
        In order to incorporate cyclicity in input data, one can add the sin/cos of the time data (e.g.).

    Args:
        df: pandas dataframe.
        col_name_time (str):  name of the column where the cyclicity is computed from.
        period: period in terms of values from the col_name_time.
        start_num (float): starting value of the cyclicity. Default = 0.

    Pre-condition:
        df's col_name_time exists.

    Post-condition:
        df's col_name_time is removed.
        df's  'sin_{col_name_time}' and 'cos_{col_name_time}' are created.

    Returns:
        The new dataframe that needs to be reassigned.
    """
    values = 2 * np.pi * (df[col_name_time] - start_num) / period
    kwargs = {f'sin_{col_name_time}': lambda x: np.sin(values),
              f'cos_{col_name_time}': lambda x: np.cos(values)}
    return df.assign(**kwargs).drop(columns=[col_name_time])
