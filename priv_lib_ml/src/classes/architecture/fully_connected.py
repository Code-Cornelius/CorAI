from abc import abstractmethod, ABCMeta

import torch.nn as nn
# my lib
from priv_lib_error import Error_type_setter
from priv_lib_util.tools import function_iterable

# Savable_net
from priv_lib_ml.src.classes.architecture.savable_net import Savable_net


# the class of NN
class Fully_connected_NN(Savable_net, metaclass=ABCMeta):
    """
    Semantics:
        Abstract classes with virtual parameters that are initialized with the function *factory_parametrised_FC_NN*
        Inherits from Savable_net that implements the concept of saving, loading, best_weights etc... allows for early stopping.

    Abstract args:
        input_size: the size of the input layer.
        list_hidden_sizes: iterable the input sizes for each hidden layer + output of last hidden layer. At least one hidden layer.
        output_size: the output size of the neural network.
        list_biases: list of booleans for specifying which layers use biases.
        activation_functions: list of activation functions for each layer. list of modules / callables.
        dropout: dropout rate for all layers. We do not dropout the first and last layer (input and output layer). In [0,1].

    Inherited :
        Args:
            best_weights:
            best_epoch:
        Class Args:
            _predict_fct
    """

    def __init__(self, predict_fct, *args, **kwargs):
        """
        Constructor for Neural Network.
        """
        super().__init__(predict_fct=predict_fct, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    #  SETTERS / GETTERS
    @property
    @abstractmethod  # ABSTRACT FIELD
    def input_size(self):
        return self._input_size

    @property
    @abstractmethod  # ABSTRACT FIELD
    def list_hidden_sizes(self):
        return self._list_hidden_sizes

    @property
    @abstractmethod  # ABSTRACT FIELD
    def output_size(self):
        return self._output_size

    @property
    @abstractmethod  # ABSTRACT FIELD
    def list_biases(self):
        return self._list_biases

    @property
    @abstractmethod  # ABSTRACT FIELD
    def activation_functions(self):
        return self._activation_functions

    @property
    @abstractmethod  # ABSTRACT FIELD
    def dropout(self):
        return self._dropout

    @property
    def nb_of_params(self):
        self._nb_of_parameters = Fully_connected_NN.compute_nb_of_params(input_size=self.input_size,
                                                                         list_hidden_sizes=self.list_hidden_sizes,
                                                                         output_size=self.output_size)
        return self._nb_of_parameters

    @staticmethod
    def compute_nb_of_params(input_size, list_hidden_sizes, output_size):
        size = input_size * list_hidden_sizes[0]

        for i in range(1, len(list_hidden_sizes)):
            size += list_hidden_sizes[i - 1] * list_hidden_sizes[i]

        size += list_hidden_sizes[-1] * output_size
        return size


    # section ######################################################################
    #  #############################################################################
    # rest of methods

    def set_layers(self):  #: mandatory call in the constructor, 
        #: to initialize all the layers and dropout with respect to the parameters created.

        # array of fully connected layers
        self._layers = nn.ModuleList()
        # initialise the input layer
        self._layers.append(nn.Linear(self.input_size, self.list_hidden_sizes[0], self.list_biases[0]))
        # initialise the hidden layers
        for i in range(len(self.list_hidden_sizes) - 1):
            self._layers.append(nn.Linear(self.list_hidden_sizes[i],
                                          self.list_hidden_sizes[i + 1],
                                          self.list_biases[i + 1]))
        # initialise the output layer
        self._layers.append(nn.Linear(self.list_hidden_sizes[-1], self.output_size, self.list_biases[-1]))
        # initialise dropout
        self._apply_dropout = nn.Dropout(p=self.dropout)

        self.init_weights_of_model()  # : init the weights in the xavier way.

    def forward(self, x):
        # pass through the input layer
        out = self.activation_functions[0](self._layers[0](x))

        # pass through the hidden layers
        for layer_index in range(1, len(self.list_hidden_sizes)):
            out = self.activation_functions[layer_index](self._apply_dropout(self._layers[layer_index](out)))

        # pass through the output layer
        out = self._layers[-1](out)
        return out




# section ######################################################################
#  #############################################################################
# CLASS FACTORY :  creates subclasses of FC NN

def factory_parametrised_FC_NN(param_input_size, param_list_hidden_sizes, param_output_size,
                               param_list_biases, param_activation_functions,
                               param_dropout=0., param_predict_fct=None):
    """
    Examples:
            input_size = 1
            hidden_sizes = [500, 500, 500, 500, 500]
            output_size = 1
            biases = [True, True, True, True, True, True]
            activation_functions = [torch.tanh, torch.tanh, torch.tanh, torch.tanh, torch.tanh]
            # example of activation functions : torch.celu, torch.relu, torch.tanh
            dropout = 0.0
            epochs = 500
            batch_size = len(rescaled_train_X)
            predict_fct = nn.Identity()
    """
    assert len(param_list_biases) == len(param_list_hidden_sizes) + 1, \
        "wrong dimensions for biases and hidden layers."

    class Parametrised_FC_NN(Fully_connected_NN):

        def __init__(self):
            self.input_size = param_input_size
            self.list_hidden_sizes = param_list_hidden_sizes
            self.output_size = param_output_size
            self.list_biases = param_list_biases  # should always be defined after list_hidden_sizes.
            self.activation_functions = param_activation_functions
            self.dropout = param_dropout

            super().__init__(predict_fct=param_predict_fct)
            self.set_layers()  #: mandatory call in the constructor,
            # :to initialize all the layers and dropout with respect to the parameters created.

        # section ######################################################################
        #  #############################################################################
        #  SETTERS / GETTERS
        @property
        def input_size(self):
            return self._input_size

        @input_size.setter
        def input_size(self, new_input_size):
            if isinstance(new_input_size, int):
                self._input_size = new_input_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def list_hidden_sizes(self):
            return self._list_hidden_sizes

        @list_hidden_sizes.setter
        def list_hidden_sizes(self, new_list_hidden_sizes):
            if function_iterable.is_iterable(new_list_hidden_sizes):
                self._list_hidden_sizes = new_list_hidden_sizes
            else:
                raise Error_type_setter(f"Argument is not an Iterable.")

        @property
        def output_size(self):
            return self._output_size

        @output_size.setter
        def output_size(self, new_output_size):
            if isinstance(new_output_size, int):
                self._output_size = new_output_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def list_biases(self):
            return self._list_biases

        # always set list_biases after list_hidden_sizes:
        @list_biases.setter
        def list_biases(self, new_list_biases):
            if function_iterable.is_iterable(new_list_biases):
                assert len(new_list_biases) == len(
                    self.list_hidden_sizes) + 1, "wrong dimensions for biases and hidden layers."
                # :security that the right parameters are given.
                self._list_biases = new_list_biases
            else:
                raise Error_type_setter(f"Argument is not an iterable.")

        @property
        def activation_functions(self):
            return self._activation_functions

        @activation_functions.setter
        def activation_functions(self, new_activation_functions):
            if function_iterable.is_iterable(new_activation_functions):
                assert len(new_activation_functions) == len(self.list_hidden_sizes), "wrong dimensions for " \
                                                                                     "activation functions"
                self._activation_functions = new_activation_functions
            else:
                raise Error_type_setter(f"Argument is not an iterable.")

        @property
        def dropout(self):
            return self._dropout

        @dropout.setter
        def dropout(self, new_dropout):
            if isinstance(new_dropout, float) and 0 <= new_dropout < 1:
                # : dropout should be a percent between 0 and 1.
                self._dropout = new_dropout
            else:
                raise Error_type_setter(f"Argument is not an {str(float)}.")



    return Parametrised_FC_NN
