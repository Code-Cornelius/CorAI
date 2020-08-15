# normal libraries
from abc import abstractmethod

import functions.tools.classical_functions_str
import numpy as np  #maths library and arrays

# my libraries
from functions.tools import classical_functions
from classes.plot.class_aplot import APlot
from classes.graphs.class_graph_estimator import Graph_Estimator

# errors:

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Histogram_estimator(Graph_Estimator):
    # abstract nb_of_bins parameter
    @property
    @abstractmethod
    def nb_of_bins(self):
        pass


    def __init__(self, estimator, separators=None):
        super().__init__(estimator = estimator, separators = separators)

    # section ######################################################################
    #  #############################################################################
    # data

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_range(self, key, mean):
        return None

    @abstractmethod
    def get_dict_param(self, key, mean):
        pass

    @abstractmethod
    def get_dict_fig(self, separators, key):
        pass

    def draw(self, separators=None):
        separators, global_dict, keys = super().draw(separators = separators)

        for key in keys:
            data = global_dict.get_group(key)['value']
            mean = data.mean()
            data = data.values
            plot = APlot()
            param_dict = self.get_dict_param(key, mean)
            fig_dict = self.get_dict_fig(separators, key)
            plot.hist(data=data, dict_param_hist=param_dict, dict_fig=fig_dict)
            name_file =  ''.join([functions.tools.classical_functions_str.tuple_to_str(key), 'histogram'])
            plot.save_plot(name_save_file=name_file)
