import pandas as pd
from priv_lib_error import Error_type_setter


# work-in-progress another idea would be to inherit from dataframes the estimator.

class Estimator(object):
    """
    SEMANTICS :
        Class Estimator as an adaptor from the dataframes from pandas.
        We use Pandas since it is fairly rampant and easy to use.
        We store the data in the following way: each column is one feature, each row one estimation.
    """

    def __init__(self, DF, *args, **kwargs):
        # args and kwargs for the super() method.
        self.DF = DF

    def __repr__(self):
        # this is for the print function. We want the estimator to inherit the properties from DF!
        return repr(self._DF)

    @classmethod
    def from_path(cls, path):
        """
        SEMANTICS :
            Constructor estimator with a path.
        Args:
            path: string. The path has to be raw, no "\".

        Returns: new estimator.

        """
        return cls(pd.read_csv(path))  # calling the constructor of the class.

    def append(self, appending_df):
        """
        SEMANTICS :
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

    def apply_function_upon_data_store_it(self, separator, fct, name, **kwargs):
        """
        SEMANTICS :
            Transform the data
            (given a whole column of data, but could be extended recursively to more than one column computations)
            with respect to a function and compute a new array of data
            that will be stored as a new column in the dataframe.

        Args:
            separator: a string, name of the feature upon which we apply the data.
            fct: function of the input data. It can be anything that takes as input a list.
            In particular, it should work for numpy array (because Pandas is related to numpy).
            name: name of the new column of the data (can be the name of an old column).
            **kwargs: any parameter that could take the function.

        Returns:
            nothing, data stored inside the given DF.
        """
        self._DF[name] = self.apply_function_upon_data(separator, fct, **kwargs)
        return

    def apply_function_upon_data(self, separator, fct, **kwargs):
        """
        SEMANTICS : Transform the data
        (given a whole column of data, but could be extended recursively to more than one column computations)
        with respect to a function and compute a new array of data which is returned.

        Args:
            separator: a string, name of the feature upon which we apply the data.
            fct: function of the input data. It can be anything that takes as input a list.
            In particular, it should work for numpy array (because Pandas is related to numpy).
            **kwargs: any parameter that could take the function.

        Returns: nothing, data stored inside the given DF.

        Examples:
            example of function, the MSE calculator:
            def compute_MSE(param, true_parameter):
                 return (param - true_parameter) ** 2
        """
        return self._DF.apply(lambda row: fct(row[separator], **kwargs), axis=1)

    def estimator_mean(self, names, separators = None):
        """
        SEMANTICS :
            empirical mean of the data separated with the keys separators at column name.
        Args:
            names: list of strings, which columns/feature are the means computed.
            separators: list of strings, which keys should be considered to groupby data together.
            If None, then no grouping by and mean computed on whole data.

         Returns: return a DF of the means.

        Dependencies :
            groupby_DF

        """
        if separators is not None:
            return self.groupby_DF(separators)[names].mean()
        else:
            return self._DF[names].mean()

    def estimator_variance(self, names, separators=None, ddof=1):
        """
        SEMANTICS : empirical variance of the data of the variance.

        Args:
            names:  list of strings, which columns/feature are the variances computed.
            separators:  list of strings, which keys should be considered to groupby data together.
            If None, then no grouping by and variance computed on whole data.
            ddof: how much one normalize the results (usually  / n-1 ;
            This gives the unbiased estimator of the variance if the mean is unknown).

        Returns: normalized S^2

        Dependencies :
            groupby_DF

        """
        if separators is not None:
            return self.groupby_DF(separators)[names].var(ddof=ddof)
        else:
            return self._DF[names].var(ddof=ddof)

    def to_csv(self, path, **kwargs):
        """
        SEMANTICS : adaptor of the method to_csv from dataframes.

        Args:
            path:  path where csv file is.
            **kwargs: kwargs for the function to_csv.

        Returns:

        """
        self._DF.to_csv(path, **kwargs)
        return

    def groupby_DF(self, separators):
        """
        SEMANTICS : groupby a DF.

        Args:
            separators: list of strings by which we groupby.

        Returns:
            tuple with the groupby as well as the keys in order to iterate over it.

        """
        DataFrameGroupBy = self._DF.groupby(separators)
        return DataFrameGroupBy, DataFrameGroupBy.groups.keys()

    @property
    def DF(self):
        # getter for df.
        return self._DF

    @DF.setter
    def DF(self, new_DF):
        # verification that the constructor is given a dataframe.
        if isinstance(new_DF, pd.DataFrame):
            self._DF = new_DF
        else:
            raise Error_type_setter('Argument is not an Dataframe.')
