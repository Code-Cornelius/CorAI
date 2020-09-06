# normal libraries
import library_functions.tools.classical_functions_vectors
import pandas as pd
# my libraries
from library_classes.estimators.class_estimator import Estimator
from library_classes.estimators.graphs.class_root_graph import Root_Graph
from library_errors.Error_type_setter import Error_type_setter
from library_functions.tools.classical_functions_vectors import is_iterable


class Graph_Estimator(Root_Graph):
    def __init__(self, estimator, separators=None, *args, **kwargs):
        """

        Args:
            estimator:  any estimator type,
            separators:  iterable type
            *args:
            **kwargs:
        """
        self.estimator = estimator
        self.separators = separators
        super().__init__(estimator=estimator, separators=separators, *args, **kwargs)

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        estimator = Estimator(pd.read_csv(path))
        return cls(estimator, None)

    # section ######################################################################
    #  #############################################################################
    # plot

    def draw(self, separators, *args, **kwargs):
        if separators is None:
            separators = self.separators
        global_dict, keys = self.estimator.groupby_DF(separators)
        return separators, global_dict, keys

    @staticmethod
    def generate_title(names, values, before_text="", extra_text=None, extra_arguments=[]):
        # extra_argument is empty list that isn't used. I don't append anything ot it or whatever.

        title = before_text
        # when a before text is given  I add to it a going to the line. Otherwise no need to jump.
        if title != "":
            title = ''.join([title, '\n'])
        list_param = [strng for strng in
                      library_functions.tools.classical_functions_vectors.roundrobin(names, [" : "] * len(values),
                                                                                     values,
                                                                                     [", "] * (len(values) - 1))]
        str_param = ''.join([str(elem) for elem in list_param])
        # list_param is including ints and str so I need to convert them all before joining,
        # since join requires only str.
        if extra_text is not None:
            # title = ''.join([title, ', ', names, ' : ', values, "\n", extra_text.format(*extra_arguments)] )
            title = ''.join([title, str_param, "\n", extra_text.format(*extra_arguments), '.'])
        else:
            title = ''.join([title, '\n', str_param, '.'])
        return title

    # section ######################################################################
    #  #############################################################################
    # data

    @staticmethod
    def test_true_value(data):
        """
        test if there is only one true value i  the given sliced data.
        It could lead to potential big errors.

        Args:
            data: sliced data from estimator.DF

        Returns:

        """
        if data['true value'].nunique() != 1:
            raise Exception(
                "Error because you are estimating different parameters, but still compounding the MSE error together.")

    # method that level up the method to csv of dataframes.
    def to_csv(self, path, **kwargs):
        self._estimator.DF.to_csv(path, **kwargs)
        return

    # section ######################################################################
    #  #############################################################################
    # the getters and setters.
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
    def separators(self):
        return self._separators

    @separators.setter
    def separators(self, new_separator):
        if is_iterable(new_separator):
            self._separators = new_separator
        else:
            raise Error_type_setter('Argument is not an iterable.')
