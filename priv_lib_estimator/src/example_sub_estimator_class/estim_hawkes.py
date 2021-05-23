# normal libraries
import pandas as pd

# my libraries
from priv_lib_estimator import Estimator
from priv_lib_error import Error_not_allowed_input


# other files


# section ######################################################################
#  #############################################################################
# class


class Estimator_Hawkes(Estimator):
    SET_COLUMN_HAWKES = {'parameter', 'n', 'm', 'time estimation',
                         'weight function', 'value', 'T_max', 'time_burn_in',
                         'true value', 'number of guesses'}

    # DF is a dataframe from pandas. Storing information inside is quite easy,
    # easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self, df=None, *args, **kwargs):
        if df is not None:
            # test that the columns of the DF are the right one, corresponding to the class argument.
            if Estimator_Hawkes.SET_COLUMN_HAWKES.issubset(df.columns):
                super().__init__(df, *args, **kwargs)
            else:
                raise Error_not_allowed_input("Problem, the columns of the dataframe do not match "
                                              "the ones from the classical estimator hawkes.")
        # if no df, we create an empty one.
        else:
            super().__init__(pd.DataFrame(columns=list(Estimator_Hawkes.SET_COLUMN_HAWKES)))

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        df = pd.read_csv(path)
        return cls(df)

    def mean(self, separator=None):
        """

        Args:
            separator: is a list, of the estimators to gather together.

        Returns: the output format is list of lists with on each line [ans_N, ans_A, ans_B],
        and on every single additional dimension, the separator.

        """

        separators = ['parameter', 'm', 'n']
        M = self.DF["m"].max() + 1
        ans_dict = {}

        # if separator is not None:
        #     for str in separator: separators.append(str)

        global_dict, keys = self.groupby_DF(separator)
        for key in keys:
            data = global_dict.get_group(key)
            dict_of_means = data.groupby(separators)['value'].mean()
            ans_N, ans_A, ans_B = [], [], []

            for i in range(M):
                ans_N.append(dict_of_means[('nu', i, 0)])
                for j in range(M):
                    if not j:  # if j == 0
                        ans_A.append([])
                        ans_B.append([])
                    # we append to this new small list the j's.
                    ans_A[i].append(dict_of_means[('alpha', i, j)])
                    ans_B[i].append(dict_of_means[('beta', i, j)])
            # i get triple list like usually.
            ans_dict[key] = [ans_N, ans_A, ans_B]
        return ans_dict
