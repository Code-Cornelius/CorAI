import pandas as pd
from priv_lib_error import Error_type_setter


# work-in-progress another idea would be to inherit from dataframes the estimator.

class Estimator(object):
    """
    SEMANTICS:
        Class Estimator as an adaptor from the dataframes from pandas.
        We use Pandas since it is fairly rampant and easy to use.
        We store the data in the following way: each column is one feature, each row one estimation.

        It is good practice to put the names of the columns / features in the class object, as a security.
    """

    NAMES_COLUMNS = {};

    def __init__(self, DF, *args, **kwargs):
        # args and kwargs for the super() method.
        self.DF = DF

    def __repr__(self):
        # this is for the print function. We want the estimator to inherit the properties from DF!
        return repr(self._DF)

    @classmethod
    def from_path_csv(cls, path):
        """
        SEMANTICS:
            Constructor estimator with a path.
        Args:
            path: string. The path has to be raw, no "\". CSV file.

        Returns: new estimator.

        """
        return cls(pd.read_csv(path))  # calling the constructor of the class.

    def append(self, appending_df):
        """
        SEMANTICS:
            adaptor for the method append from DF at the estimator level.

        Args:
            appending_df: the df to append to the self object.

        Returns:
            quoting : "Pandas dataframe.append() function
            is used to append rows of other dataframe to the end of the given dataframe,
            returning a new dataframe object.
            Columns not in the original dataframes are added as new columns
            and the new cells are populated with NaN value."

        REFERENCES : https://www.geeksforgeeks.org/python-pandas-dataframe-append/
        """
        self._DF = self._DF.append(appending_df)

    def apply_function_upon_data(self, separators, fct, **kwargs):
        """
        SEMANTICS : Transform the data
        with respect to a function and compute a new array of data which is returned.
        Will do: fct(DF[separators]) component wise.

        Args:
            separators: a string, name of the feature upon which we apply the data.

            fct: function of the input data. It can be anything that can be applied component wise.
            In particular, it should work for numpy array (because Pandas is related to numpy).

            **kwargs: Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns: returns the transformed slice dataframe.

        Examples:
            example of function, the MSE calculator:
            def compute_MSE(param, true_parameter):
                 return (param - true_parameter) ** 2
            or any function from numpy, like np.sqrt
        """

        # trick for applying the function on a slice (column slice) of the data.
        return self._DF.apply(lambda row: fct(row[separators], **kwargs), axis=1)

    def apply_function_upon_data_store_it(self, separators, fct, new_column_names, **kwargs):
        """
        SEMANTICS:
            Transform the data
            (given a whole column of data, but could be extended recursively to more than one column computations)
            with respect to a function and compute a new array of data
            that will be stored as a new column in the dataframe.

        Args:
            separators: list of strings, name of the features upon which we apply the data.

            fct: function of the input data. It can be anything that can be applied component wise.
            In particular, it should work for numpy array (because Pandas is related to numpy).

            new_column_names: list of the names of the new columns of the data (can be the name of an old column).
            the new_column_names has the same length as separators, (see apply_function_upon_data return)

            **kwargs: Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns:
            nothing, data stored inside the given DF.

        Dependencies:
            apply_function_upon_data
        """
        assert len(new_column_names) == len(separators), "New_column_names and separators must have same dimension."
        self._DF[new_column_names] = self.apply_function_upon_data(separators, fct, **kwargs)
        return

    def estimation_group_mean(self, columns_for_computation, keys_grouping=None):
        """
        SEMANTICS:
            empirical mean of the data separated with the keys keys_grouping at column name.
        Args:
            columns_for_computation: list of strings, which columns/feature are the means computed.
            keys_grouping: list of strings, which keys should be considered to groupby data together.
            If None, then no grouping by and mean computed on whole data.

         Returns: return a DF of the means.

        Dependencies:
            groupby_DF

        """
        if keys_grouping is None:
            return self._DF[columns_for_computation].mean()
        else:
            return self.groupby_DF(keys_grouping)[0][columns_for_computation].mean()
            #                      keys are how we groupby
            #                                    [0] because groupby hands back a tuple and we need the groups
            #                                        which feature are we interested in.

    def estimation_group_variance(self, columns_for_computation, keys_grouping=None, ddof=1):
        """
        SEMANTICS: empirical variance of the data of the variance.

        Args:
            columns_for_computation:  list of strings, which columns/feature are the variances computed.
            keys_grouping:  list of strings, which keys should be considered to groupby data together.
            If None, then no grouping by and variance computed on whole data.
            ddof: how much one normalize the results (usually  / how_much_rotate-1 ;
            This gives the unbiased estimator of the variance if the mean is unknown).

        Returns: normalized S^2

        Dependencies:
            groupby_DF

        """
        if keys_grouping is not None:
            return self.groupby_DF(keys_grouping)[columns_for_computation].var(ddof=ddof)
        else:
            return self._DF[columns_for_computation].var(ddof=ddof)

    def to_csv(self, path, **kwargs):
        """
        SEMANTICS :
            void adaptor of the method to_csv from dataframes.

        Args:
            path: path where the dataframe of the estimator is saved.
            **kwargs: Additional keyword arguments to pass as keywords arguments to
            the function to_csv of pandas.

        Returns:

        """
        self._DF.to_csv(path, **kwargs)
        return

    def groupby_DF(self, separators, order=True):
        """
        SEMANTICS :
            groupby a DF.

        Args:
            separators: list of strings by which we groupby.
            order: determines whether an ordering is done or not.

        Returns:
            tuple with the groupby as well as the keys in order to iterate over it.

        """
        DataFrameGroupBy = self._DF.groupby(separators, order)
        return DataFrameGroupBy, DataFrameGroupBy.groups.keys()

    @property
    def DF(self):
        # getter for df.
        return self._DF

    @DF.setter
    def DF(self, new_DF):
        # verification that the constructor is given a pandas dataframe.
        if isinstance(new_DF, pd.DataFrame):
            self._DF = new_DF
        else:
            raise Error_type_setter('Argument is not an Dataframe.')
