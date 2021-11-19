from abc import ABCMeta

import torch
import torch.nn as nn

from corai.src.classes.architecture.rnn.rnn import RNN


class Two_hidden_recurrent(RNN):
    def __init__(self):
        super().__init__()
        #h0 and c0
        self.hidden_state_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                       1,  # repeated later to have batch size
                                                       self.hidden_size),
                                           requires_grad=True)  # parameters are moved to device and learn.

        self.hidden_cell_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                      1,  # repeated later to have batch size
                                                      self.hidden_size),
                                          requires_grad=True)  # parameters are moved to device and learn.

    def get_hidden_states(self, batch_size):
        return self.hidden_state_0.repeat(batch_size), self.hidden_cell_0.repeat(batch_size)