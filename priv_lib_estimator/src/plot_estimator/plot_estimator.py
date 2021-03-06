# normal libraries
import pandas as pd
from abc import abstractmethod

# my libraries
from priv_lib_estimator.src.plot_estimator.root_plot_estimator import Root_plot_estimator
from priv_lib_error import Error_type_setter
from priv_lib_estimator.src.estimator.estimator import Estimator
from priv_lib_util.tools import function_iterable


class Plot_estimator(Root_plot_estimator):
    """
    Semantics:
        Plot_estimator is an abstract class that abstract the idea of plotting
        an estimator (class defined in priv_lib_estimator.src.estimator.estimator).
        The aim is to define the basement of what is required to plot an estimator:
            - an Estimator: the object giving the data,
            - a slicer: which part of the data we plot,
            - a figure: on what we plot.
        The design choice that has been done is
        a plot estimator HAS an estimator that will be plotted.

    """

    def __init__(self, estimator, grouping_by=None, *args, **kwargs):
        """
        Args:
            estimator:  object from priv_lib_estimator.src.estimator.estimator
            grouping_by:  list of names of features from estimator.
            Default is None, and means no grouping is done, so all data is used for plotting.

            *args: additional parameters for the construction of the object.
            **kwargs: additional parameters for the construction of the object.
        """
        self.estimator = estimator
        self.grouping_by = grouping_by
        super().__init__(estimator=estimator, separators=grouping_by, *args, **kwargs)

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
    def draw(self, separators=None, *args, **kwargs):
        """
        Semantics:
            drawing method for plotting the results.
        Args:
            separators: if None given, the separators are the one of the object given at creation.
            *args:
            **kwargs:

        Returns:

        """
        if separators is None:
            separators = self.grouping_by
        global_dict, keys = self.estimator.groupby_DF(separators)
        return separators, global_dict, keys

    @staticmethod
    def generate_title(parameters, parameters_value, before_text="", extra_text=None, extra_arguments=[]):
        """
        Semantics:
            generate a title given the parameters. Essentially, the title looks like:
                [before_text \n]
                names[0]: values[0], ... names[n]: values[n]
                [\n extra_text.format(*extra_arguments)]
                .

        Args:
            parameters: list of parameter we wish to put in the title
            parameters_value: list of the values of the previous parameters
            before_text: string
            extra_text: string with some holes for extra_arguments.
            extra_arguments: the arguments to add to extra_text.

        Returns:
            the title.

        Examples:
             generate_title(["$sigma$"], [3], before_text="The title", extra_text=None, extra_arguments=[])
             ->
             generate_title(["$sigma$", "\rho"], [3,0.5], before_text="", extra_text=None, extra_arguments=[])
             ->
             generate_title(["$sigma$", "\rho"], [3,0.5], before_text="", extra_text="first param. is {} and second {}.", extra_arguments=[3.000, 62])
             ->

        """
        assert len(parameters) == len(parameters_value), "Parameters and parameters_value should have the same length."
        beg_title_with_new_line = before_text
        # when a before text is given  I add to it a going to the line. Otherwise no need to jump.
        if beg_title_with_new_line != "":
            beg_title_with_new_line = ''.join([beg_title_with_new_line, '\n'])
        list_param = [strng for strng in function_iterable.
            roundrobin(parameters,
                       [": "] * len(parameters_value),
                       parameters_value,
                       [", "] * (len(parameters_value) - 1))
                      ]
        names_and_values = ''.join([str(elem) for elem in list_param])
        # list_param is including ints and str so I need to convert them all before joining,
        # since join requires only str.

        if extra_text is not None:
            title = ''.join([beg_title_with_new_line, names_and_values, "\n", extra_text.format(*extra_arguments), '.'])
        else:
            title = ''.join([beg_title_with_new_line, names_and_values, '.'])
        return title

    # section ######################################################################
    #  #############################################################################
    # data

    @staticmethod
    def test_true_value(data):
        # TODO is it useful?
        """

        test if there is only one true value in the given sliced data.
        It could lead to potential big errors.

        Args:
            data: sliced data from estimator.DF

        Returns:

        """
        if data['true value'].nunique() != 1:
            raise Exception(
                "Error because you are estimating different parameters, but still compounding the MSE error together.")

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
        if function_iterable.is_iterable(new_grouping_by):
            # TODO test whether the new_grouping_by is
            #  a subset of the columns of the Estimator.
            self._grouping_by = new_grouping_by
        else:
            raise Error_type_setter('Argument is not an iterable.')
