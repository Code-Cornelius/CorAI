import pandas as pd

class Estimator:
    # DF is a dataframe from pandas. Storing information inside is quite easy, easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self, DF):
        self.DF = DF

    def __repr__(self):
        return repr(self.DF)

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        return cls(pd.read_csv(path))


    def append(self, new_df):
        '''
        redefine the method append from DF at the estimator level.

        Args:
            new_df:

        Returns:

        '''
        self.DF = (self.DF).append(new_df)

    def function_upon_separated_data(self, separator, fct, name, **kwargs):
        # separator is a string
        # fct is a fct
        # name is the name of a column where the data will lie.
        # one value is one parameter... is it enough parameter ?
        # the function does create a new column in the DF, by looking at the data in the separator and applying the function to it.
        self.DF[name] = self.DF.apply(lambda row: fct(row[separator], **kwargs), axis=1)
        return

    def mean(self, name, separators = None):
        ## name is the name of a column where the data lies.
        if separators is not None:
            return self.DF.groupby(separators)[name].mean()
        else :
            return self.DF[name].mean()


    # it corresponds to S^2. This is the empirical estimator of the variance.
    def estimator_variance(self, name, separators = None, ddof=1):
        ## ddof is by how much one normalize the results (usually  / n-1). This gives the unbiased estimator of the variance if the mean is known.
        if separators is not None:
            return self.DF.groupby(separators)[name].var(ddof=ddof)
        else :
            return self.DF[name].var(ddof=ddof)

    # method that level up the method to csv of dataframes.
    def to_csv(self, path, **kwargs):
        self.DF.to_csv(path, **kwargs)
        return

    def groupby_DF(self, separators):
        '''
        groupby a DF.

        Args:
            separators:

        Returns:
            tuple with the groupby as well as the keys in order to iterate over it.

        '''
        dictionary = self.DF.groupby(separators)
        return dictionary, dictionary.groups.keys()