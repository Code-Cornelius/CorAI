from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from corai_error import Error_type_setter

from corai.src.classes.architecture.rnn.rnn import RNN
from corai.src.classes.architecture.savable_net import Savable_net


class One_hidden_recurrent(RNN, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        # h0
        self.hidden_state_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                       1,  # repeated later to have batch size
                                                       self.hidden_size),
                                           requires_grad=True)  # parameters are moved to device and learn.



    def get_hidden_states(self, batch_size):
        return self.hidden_state_0.repeat(batch_size)
