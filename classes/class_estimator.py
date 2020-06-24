
class Estimator:
    # DF is a dataframe from pandas. Storing information inside is quite easy, easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self, DF):
        self.DF = DF

    def __repr__(self):
        return repr(self.DF)

    def append(self, new_df):
        self.DF = (self.DF).append(new_df)

    def function_upon_separeted_data(self, separator, fct, name, **kwargs):
        # separator is a string
        # fct is a fct
        # name is the name of a column where the data will lie.
        # one value is one parameter... is it enough parameter ?
        # the function does create a new column in the DF, by looking at the data in the separator and applying the function to it.
        self.DF[name] = self.DF.apply(lambda row: fct(row[separator], **kwargs), axis=1)
        return

    def mean(self, name, separators):
        ## name is the name of a column where the data lies.
        return self.DF.groupby(separators)[name].mean()

    # it corresponds to S^2. This is the empirical estimator of the variance.
    def estimator_variance(self, name, ddof=1):
        ## ddof is by how much one normalize the results (usually  / n-1). This gives the unbiased estimator of the variance if the mean is known.
        return self.DF[name].var(ddof=ddof)

    def to_csv(self, path, **kwargs):
        self.DF.to_csv(path, **kwargs)
        return

    def slice_DF(self, separators):
        dictionary = self.DF.groupby(separators)
        return dictionary, dictionary.groups.keys()