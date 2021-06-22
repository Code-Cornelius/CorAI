# normal libraries
from abc import abstractmethod

import pandas as pd
from priv_lib_error import Error_type_setter
from priv_lib_estimator.src.estimator.estimator import Estimator
# my libraries
from priv_lib_estimator.src.plot_estimator.root_plot_estimator import Root_plot_estimator
from priv_lib_plot import AColorsetDiscrete
from priv_lib_util.tools import function_iterable


class Plot_estimator(Root_plot_estimator):
    """
    Semantics:
        Plot_estimator is an abstract class that abstract the idea of plotting
        an estimator (class defined in priv_lib_estimator.src.estimator.estimator).
        The aim is to define the base of what is required to plot an estimator:
            - an Estimator: the object giving the data,
            - a slicer: which part of the data we plot,
            - a figure: on what we plot.
        The design choice that has been done is
        a plot estimator HAS an estimator that will be plotted.

        One can use generate_title for creating the title of figures.
        Overwrite draw.

    """
    COLORMAP = AColorsetDiscrete('Dark2')  # colormap for all plots.

    # it can be changed by self.COLORMAP; setting it allows to always use the same colormaps.

    def __init__(self, estimator, grouping_by=None, *args, **kwargs):
        """
        Args:
            estimator (priv_lib_estimator.src.estimator.estimator):
            grouping_by (iter of str):  list of names of features from estimator.
            Default is None, and means no grouping is done, so all data is used for plotting.

            *args: additional parameters for the construction of the object.
            **kwargs: additional parameters for the construction of the object.
        """
        self.estimator = estimator
        self.grouping_by = grouping_by
        super().__init__(estimator=estimator, grouping_by=grouping_by, *args, **kwargs)

    @classmethod
    def from_path_csv(cls, path, grouping_by=None):
        """
        Semantics:
            Constructor plot_estimator with a path.
        Args:
            path: string. The path has to be raw, no "\". CSV file.
            grouping_by: iterable of features.

        Returns: new plot_estimator.

        """
        estimator = Estimator(pd.read_csv(path))
        return cls(estimator, grouping_by)

    # section ######################################################################
    #  #############################################################################
    # plotting methods

    @abstractmethod
    def draw(self, separators_plot=None, not_use_grouping_by=False, *args, **kwargs):
        """
        Semantics:
            drawing method for plotting the results.
        Args:
            separators (list of str): Columns to split the dataframe upon. It will be merged with the grouping_by of
                the estimator
            not_use_grouping_by (bool): if true, then draw does not separate wrt self.grouping_by.

        Returns:
                - The separators (the separators received as input together with the grouping_by)
                - The global_dict which is a grouping based on the separators or the complete dataframe if
                    no separators are present
                - The keys (iterable) representing the unique identifiers for each group
        """
        if separators_plot is None:  # separators is either a list or None
            if not_use_grouping_by:
                separators_plot = []
            else:
                separators_plot = list(self.grouping_by)
        # separators not None
        else:
            separators_plot = separators_plot
            if not not_use_grouping_by:
                separators_plot += list(self.grouping_by)

        if len(separators_plot):  # >=1:
            global_dict, keys = self.estimator.groupby(separators_plot)
        else:  # =0
            global_dict, keys = self.estimator.df, [None]  # : keys is a list with None,
            # : it will be understood outside as take the whole estimator and do not use get_group.
            # : we do that because there is no way to use groupby to have a single groupbyDataframe.

        return separators_plot, global_dict, keys

    @staticmethod
    def generate_title(parameters, parameters_value, before_text='', extra_text=None, extra_arguments=[]):
        """
        Semantics:
            generate a title given the parameters. Essentially, the title looks like:
                [before_text \n]  (optional)
                names[0]: values[0], ..., names[n]: values[n]
                [\n extra_text.format(*extra_arguments)].  (optional)

            Be careful with special characters.

        Args:
            parameters: list of parameter we wish to put in the title.
            parameters_value: list of the values of the previous parameters.
            before_text: string.
            extra_text: string with some holes for extra_arguments.
            extra_arguments: the arguments to add to extra_text.

        Returns:
            a string, the title.

        Examples:
             generate_title(["$sigma$"], [3], before_text="The title", extra_text=None, extra_arguments=[])
             -> The title
                sigma : 3.
             generate_title(["$sigma$", "\rho"], [3,0.5], before_text="", extra_text=None, extra_arguments=[])
             -> sigma : 3, rho : 0.5.
             generate_title(["$sigma$", "\rho"], [3,0.5], before_text="", extra_text="first param. is {} and second {}", extra_arguments=[3.000, 62])
             -> sigma : 3, rho : 0.5
                first param. is 3.000 and second 62.

        """
        assert len(parameters) == len(parameters_value), "Parameters and parameters_value should have the same length."
        beg_title_with_new_line = before_text
        # when a before text is given,  carriage return. Otherwise no need for a newline.
        if beg_title_with_new_line != '':
            beg_title_with_new_line = ''.join([beg_title_with_new_line, '\n'])

        # put parameters and parameters value together.
        list_param = [strng for strng in function_iterable.roundrobin(parameters,
                                                                      [": "] * len(parameters_value),
                                                                      parameters_value,
                                                                      [", "] * (len(parameters_value) - 1))
                      ]
        names_and_values = ''.join([str(elem) for elem in list_param])
        # : list_param is including floats and str so I need to convert them all before joining,
        # since join handles only str.

        if extra_text is not None:
            title = ''.join([beg_title_with_new_line, names_and_values, "\n", extra_text.format(*extra_arguments), '.'])
        else:
            title = ''.join([beg_title_with_new_line, names_and_values, '.'])
        return title

    # section ######################################################################
    #  #############################################################################
    # testing conditions

    @staticmethod
    def is_true_value_unique(data):
        # TODO is it useful?
        """
        Semantics:
            Test if there is only one true value in the given data.
            The test should be done every time one uses true_value. It is
            a routine check such that the program fails hard and fast.

        Args:
            data (dataframe): the data where true value lies.

        Returns:

        """
        if data['true value'].nunique() != 1:
            raise Exception("Error because you are estimating different parameters, "
                            "but still compounding the MSE error together.")

    def is_grouping_by_subset_columns(self, grouping_by):
        return set(grouping_by).issubset(self.estimator.columns)

    # section ######################################################################
    #  #############################################################################
    # getters and setters.
    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, new_estimator):
        if isinstance(new_estimator, Estimator):
            self._estimator = new_estimator
        else:
            raise Error_type_setter('Argument is not an estimator.')

    @property
    def grouping_by(self):
        return self._grouping_by

    @grouping_by.setter
    def grouping_by(self, new_grouping_by):
        # checks if the argument is a str, an iterable or is not a subset of the columns.
        if new_grouping_by is None:
            self._grouping_by = ()  # empty tuple, which is an iterable as required
            return

        if isinstance(new_grouping_by, str):  # strings are iterables.
            raise Error_type_setter('grouping_by should be a list and not a string.')

        if not function_iterable.is_iterable(new_grouping_by):
            raise Error_type_setter('grouping_by is not an iterable.')

        if self.is_grouping_by_subset_columns(new_grouping_by):
            self._grouping_by = new_grouping_by
        else:
            raise Error_type_setter('grouping_by is not a subset of the dataframe given in estimator.')
