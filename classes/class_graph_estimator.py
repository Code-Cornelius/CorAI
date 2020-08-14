# normal libraries
from abc import abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my libraries
from classes.class_estimator import Estimator
from plot_functions import APlot
import classical_functions
from classes.class_root_graph import Root_Graph



class Graph_Estimator(Root_Graph):
    def __init__(self, estimator, separators=None):
        self.estimator = estimator
        self.separators = separators
        super().__init__()

    @classmethod
    def from_path(cls, path):
        # path has to be raw. with \\
        estimator = Estimator(pd.read_csv(path))
        return cls(estimator, None)

    def generate_title(self, names, values, before_text = "", extra_text=None, extra_arguments=[]):
        # extra_argument is empty list that isn't used. I don't append anything ot it or whatever.

        title = before_text
        #when a before text is given  I add to it a going to the line. Otherwise no need to jump.
        if title != "":
            title = ''.join([title, '\n'])
        list_param = [str for str in classical_functions.roundrobin(names, [" : "]*len(values), values, [", "]*(len(values)-1)  ) ]
        str_param = ''.join([str(elem) for elem in list_param]) # list_param is including ints and str so I need to convert them all before joining, since join requires only str.
        if extra_text is not None:
            # title = ''.join([title, ', ', names, ' : ', values, "\n", extra_text.format(*extra_arguments)] )
            title = ''.join([title, str_param, "\n", extra_text.format(*extra_arguments), '.'])
        else:
            title = ''.join([title, '\n', str_param, '.'])
        return title



    def test_true_value(self, data):
        '''
        test if there is only one true value i  the given sliced data.
        It could lead to potential big errors.

        Args:
            data: sliced data from estimator.DF

        Returns:

        '''
        if data['true value'].nunique() != 1:
            raise ("Error because you are estimating different parameters, but still compounding the MSE error together.")

    # method that level up the method to csv of dataframes.
    def to_csv(self, path, **kwargs):
        self.estimator.DF.to_csv(path, **kwargs)
        return