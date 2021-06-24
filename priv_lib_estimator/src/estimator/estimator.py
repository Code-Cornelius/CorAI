import pandas as pd
from priv_lib_error import Error_type_setter
import json

from priv_lib_util.tools.src.function_json import zip_json, unzip_json


class Estimator(object):
    """
    Semantics:
        Class Estimator is an adaptor from the dataframes from pandas.
        We use Pandas since it is fairly rampant and easy to use.
        We store the data in the following way: each column is one feature, each row one estimation.

        The objective of it is to automatise some behavior one could have regarding some containers.
        For example, always plotting the same objects in the same way, or the same estimates.
        Addit. to dataframes, estimator also allows to store additional information.
        Currently, pandas does not offer such functionality.

        For this reason, Estimator is a common ground for all derived class from Estimator.

        It is good practice to put the names of the columns / features in the class object, as a security.
    """

    CORE_COL = set()

    def __init__(self, df=None, *args, **kwargs):
        # args and kwargs for the child super() method.
        if df is not None:
            # test that the columns of the df are the right one, corresponding to the class argument.
            # the fact that we use self. ensures that we use polymorphism.
            self.df = df  # we do that first to check that it is a dataframe.
            if self.CORE_COL.issubset(df.columns):
                super().__init__()
            else:
                raise Error_type_setter("Problem, the columns of the dataframe do not match the predefined ones.")
        # if no df, we create an empty one.
        else:
            self.df = pd.DataFrame(columns=list(self.CORE_COL))
            super().__init__()

    def __repr__(self):
        # this is for the print function.
        # We want the estimator to inherit the properties from DF!
        return repr(self.df)

    @classmethod
    def from_path_csv(cls, path):
        """
        Semantics:
            Constructor estimator with a path.
        Args:
            path: string, path to a CSV file/txt.

        Returns: new estimator.

        """
        return cls(df = pd.read_csv(path))  # calling the constructor of the class.

    @classmethod
    def from_json(cls, path):
        """
            Read json dataframe an return the object
        Args:
            path: The path where to retrieve the dataframe from

        Returns:
            Void
        """
        dataframe = pd.read_json(path, orient='split')
        return cls(df = dataframe)

    @staticmethod
    def from_json_attributes(path, compress):
        # TODO 24/06/2021 nie_k:  explain how to load and save an estimator.
        #  Apparently, some functions need to be overriden... explain the process at the right place.
        #  and say where an example can be found.
        """
            Retrieve extra attributes from the json and write it back to the file
        Args:
            path: The path to the file
            compress: Whether or not compression is applied

        Returns:

        """
        with open(path, 'r') as file:
            df_info = json.load(file)
            if compress:
                df_info = unzip_json(df_info)
            attrs = df_info['attrs']
            del df_info['attrs']

        with open(path, 'w') as file:
            json.dump(df_info, file)

        return attrs
    
    # section ######################################################################
    #  #############################################################################
    # Methods
    

    def append(self, appending_df, *args, **kwargs):
        """
        Semantics:
            adaptor for the method append from DF at the estimator level.

        Args:
            appending_df: the df to append to the self object.

        Returns:
            quoting: "Pandas dataframe.append() function
            is used to append rows of other dataframe to the end of the given dataframe,
            returning a new dataframe object. I.E. NOT IN PLACE.
            Columns not in the original dataframes are added as new columns
            and the new cells are populated with NaN value."

        References: https://www.geeksforgeeks.org/python-pandas-dataframe-append/
        """
        self.df = self.df.append(appending_df, *args, **kwargs)
        self.df.reset_index(drop=True, inplace=True)  # Ensure uniqueness of the indices

    def apply_function_upon_data(self, separators, fct, **kwargs):
        # TODO verify it does what one wants.
        """
        Semantics: Transform the data
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
        return self.df.apply(lambda row: fct(row[separators], **kwargs), axis=1)

    def apply_function_upon_data_store_it(self, separators, fct, new_column_names, **kwargs):
        # TODO verify it does what one wants.
        """
        Semantics:
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
        self.df[new_column_names] = self.apply_function_upon_data(separators, fct, **kwargs)
        return

    # def estimation_group_mean(self, columns_for_computation, keys_grouping=None):
    #     # TODO verify it does what one wants.
    #     """
    #     Semantics:
    #         empirical mean of the data separated with the keys keys_grouping at column name.
    #     Args:
    #         columns_for_computation: list of strings, which columns/feature are the means computed.
    #         keys_grouping: list of strings, which keys should be considered to groupby data together.
    #         If None, then no grouping by and mean computed on whole data.
    #
    #      Returns: return a df of the means.
    #
    #     Dependencies:
    #         groupby_DF
    #
    #     """
    #     if keys_grouping is None:
    #         return self.df[columns_for_computation].mean()
    #     else:
    #         return self.groupby_DF(keys_grouping)[0][columns_for_computation].mean()
    #         #                      keys are how we groupby
    #         #                                    [0] because groupby hands back a tuple and we need the groups
    #         #                                        for which feature are we interested in.
    #
    # def estimation_group_variance(self, columns_for_computation, keys_grouping=None, ddof=1):
    #     # TODO verify it does what one wants.
    #     """
    #     Semantics: empirical variance of the data of the variance.
    #
    #     Args:
    #         columns_for_computation:  list of strings, which columns/feature are the variances computed.
    #         keys_grouping:  list of strings, which keys should be considered to groupby data together.
    #             If None, then no grouping by and variance computed on whole data.
    #         ddof: delta of degree of freedom, how much one normalize the results
    #             (usually  you divide by (len data-1), this gives the
    #             unbiased estimator of the variance if the mean is unknown).
    #
    #     Returns: normalized S^2
    #
    #     Dependencies:
    #         groupby_DF
    #
    #     """
    #     if keys_grouping is not None:
    #         return self.groupby_DF(keys_grouping)[columns_for_computation].var(ddof=ddof)
    #     else:
    #         return self.df[columns_for_computation].var(ddof=ddof)


    # section ######################################################################
    #  #############################################################################
    # Save methods
    
    def to_csv(self, path, **kwargs):
        """
        Semantics:
            void adaptor of the method to_csv from dataframes.
            Does not save the attributes. For this, use to_json.

        Args:
            path: path where the dataframe of the estimator is saved. Extension should be written.
            **kwargs: Additional keyword arguments to pass as keywords arguments to
            pandas' function to_csv.

        Returns:

        """
        self.df.to_csv(path, **kwargs)
        return

    def to_json(self, path, compress=True, attrs={}.copy()):
        # TODO 24/06/2021 nie_k: work on it
        """
            Save an estimator to json as a compressed file.
        Args:
            attrs: The extra attributes to save
            compress: Whether or not compression is applied
            path: The path where to store the estimator, with extension.

        Returns:
            Void
        """
        json_df = self.df.to_json(orient='split')
        parsed = json.loads(json_df)
        parsed['attrs'] = attrs

        if compress:
            parsed = zip_json(parsed)

        with open(path, 'w') as file:
            json.dump(parsed, file)



    @staticmethod
    def groupby_data(data, separators):
        """
        Semantics:
            groupby the dataframe and retrieve the keys.

        Args:
            data (DataFrame): dataframe to groupby.
            separators: list of strings by which we groupby.

        Returns:
            tuple with the groupby as well as the keys in order to iterate over it.

        """
        DataFrameGroupBy = data.groupby(separators)
        return DataFrameGroupBy, DataFrameGroupBy.groups.keys()

    def groupby(self, separators):
        """
        Semantics:
            groupby a DF.

        Args:
            separators: list of strings by which we groupby.

        Returns:
            tuple with the groupby as well as the keys in order to iterate over it.

        Dependency:
            groupby_data (static)
        """
        return Estimator.groupby_data(self.df, separators)

    def contains(self, column):
        return column in self.df.columns

    @property
    def df(self):
        # getter for df.
        return self._df

    @df.setter
    def df(self, new_df):
        # verification that the constructor is given a pandas dataframe.
        if isinstance(new_df, pd.DataFrame):
            self._df = new_df
        else:
            raise Error_type_setter('Argument is not an Dataframe.')

    @property
    def columns(self):
        return self.df.columns
