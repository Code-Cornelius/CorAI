from abc import ABCMeta, abstractmethod

import torch

from corai.src.classes.architecture.savable_net import Savable_net
from corai_error import Error_type_setter


class RNN(Savable_net, metaclass=ABCMeta):
    """
    One can construct from factory_parameterised_rnn.
    The forward takes only one input, time_series and h0 being together in a tuple.
    """

    def __init__(self):
        super().__init__(predict_fct=None)  # predict is identity

        self.nb_directions = int(self.bidirectional) + 1

        self.stacked_rnn = self.rnn_class(self.input_dim, self.hidden_size,
                                          num_layers=self.num_layers,
                                          dropout=self.dropout,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)

        self.output_len = self.hidden_size * self.nb_directions * self.nb_output_consider  # dim of output of forward.

    def forward(self, time_series_and_h0):
        """
        Args:
            packed together in a tuple:
                time_series (tensor): shape batch size N,  Length sequence, Linput_dim
                h0 (tensor): shape (self.num_layers * self.nb_directions, 1, self.hidden_size)
            The reason why the two parameters are packed together is in order to
            be able to associate RNN with free_nn, which only accepts functions of 1 input.
            If some other applications require to have two inputs,
            just add an adaptor layer (not the most efficient way of doing it indeed).

        Examples:
            # WIP This needs to be updated, poorly written.
            h0 can be created as:
            batch_size = (1,batch_nb,1)
            hidden_state_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                       1,  # repeated later to have batch size
                                                       self.hidden_size),
                                           requires_grad=True)  # parameters are moved to device and learn.
            hidden_state_0.repeat(batch_size)
        Returns:
        """
        time_series, h0 = time_series_and_h0  # unpacking
        out, _ = self.stacked_rnn(time_series, h0)  # shape of out is  N,L,Hidden_size * nb_direction

        if self.bidirectional:
            out = torch.cat((out[:, -self.nb_output_consider:, :self.hidden_size],
                             out[:, :self.nb_output_consider, self.hidden_size:]), 1)
            # this is where the output lies. We take nb_output elements. Meaning the h_n, h_n-1...
            # the shape of out at this stage is (N,  nb_output_consider, Hidden_size * nb_direction)

            # we do this because when the output is bidirectional, one should consider different outputs.

            # the first item is the uni direct, on top of it is stacked the other dirctn, whose first elmnts are taken.
        else:
            out = out[:, -self.nb_output_consider:, :self.hidden_size]
        return out  # shape is (batch size, nb_output_consider, hidden_size)

    # section ######################################################################
    #  #############################################################################
    # SETTERS GETTERS

    @property
    @abstractmethod
    def input_dim(self):
        return self._input_dim

    @property
    @abstractmethod
    def output_dim(self):
        return self._output_dim

    @property
    @abstractmethod
    def hidden_size(self):
        return self._hidden_size

    @property
    @abstractmethod
    def bidirectional(self):
        return self._bidirectional

    @property
    @abstractmethod
    def num_layers(self):
        return self._num_layers

    @property
    @abstractmethod  # ABSTRACT FIELD
    def dropout(self):
        return self._dropout

    @property
    @abstractmethod
    def nb_output_consider(self):
        return self._nb_output_consider

    @property
    @abstractmethod
    def rnn_class(self):
        return self._nn_class


def factory_parametrised_RNN(input_dim=1, output_dim=1, num_layers=1, bidirectional=False, nb_output_consider=1,
                             hidden_size=150, dropout=0., *, rnn_class):
    """
    GRU and LSTM are very close in terms of architecture.
    This factory allows to construct one or the other at will.

    Args:
        input_dim:
        output_dim:
        num_layers:
        bidirectional:
        nb_output_consider:
        hidden_size:
        dropout: In [0,1].
        activation_fct:
        hidden_FC:
        rnn_class: module where the parameters are. Can be:
            https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
            https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU

    Returns:

    """

    class Parametrised_RNN(RNN):
        def __init__(self):
            self.input_dim = input_dim
            self.output_dim = output_dim

            self.nb_output_consider = nb_output_consider

            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.rnn_class = rnn_class
            super().__init__()
            # https://stackoverflow.com/questions/43080583/attributeerror-cannot-assign-module-before-module-init-call

        # section ######################################################################
        #  #############################################################################
        # SETTERS GETTERS

        @property
        def input_dim(self):
            return self._input_dim

        @input_dim.setter
        def input_dim(self, new_input_dim):
            if isinstance(new_input_dim, int):
                self._input_dim = new_input_dim
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def output_dim(self):
            return self._output_dim

        @output_dim.setter
        def output_dim(self, new_output_dim):
            if isinstance(new_output_dim, int):
                self._output_dim = new_output_dim
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def hidden_size(self):
            return self._hidden_size

        @hidden_size.setter
        def hidden_size(self, new_hidden_size):
            if isinstance(new_hidden_size, int):
                self._hidden_size = new_hidden_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def bidirectional(self):
            return self._bidirectional

        @bidirectional.setter
        def bidirectional(self, new_bidirectional):
            if isinstance(new_bidirectional, bool):
                self._bidirectional = new_bidirectional
            else:
                raise Error_type_setter(f"Argument is not an {str(bool)}.")

        @property
        def num_layers(self):
            return self._num_layers

        @num_layers.setter
        def num_layers(self, new_num_layers):
            if isinstance(new_num_layers, int):
                self._num_layers = new_num_layers
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def dropout(self):
            return self._dropout

        @dropout.setter
        def dropout(self, new_dropout):
            if isinstance(new_dropout, float) and 0 <= new_dropout < 1:
                # : dropout should be a percent between 0 and 1.
                self._dropout = new_dropout
            else:
                if isinstance(new_dropout, int) and not (new_dropout):  # dropout == 0
                    self._dropout = float(new_dropout)
                else:
                    raise Error_type_setter(f"Argument is not an {str(float)}.")


        @property
        def nb_output_consider(self):
            return self._nb_output_consider

        @nb_output_consider.setter
        def nb_output_consider(self, new_nb_output_consider):
            if isinstance(new_nb_output_consider, int):
                self._nb_output_consider = new_nb_output_consider
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def rnn_class(self):
            return self._rnn_class

        @rnn_class.setter
        def rnn_class(self, new_nn_class):
            self._rnn_class = new_nn_class

    return Parametrised_RNN
