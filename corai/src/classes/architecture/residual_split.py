from abc import abstractmethod, ABCMeta

import torch.nn as nn
# my lib
from priv_lib_error import Error_type_setter
from priv_lib_util.tools import function_iterable

# Savable_net
from priv_lib_ml.src.classes.architecture.savable_net import Savable_net

class Residual_split(Savable_net):
    def __init__(self, in_p, out_p):
        super().__init__(None)
        self.first_layer = nn.Linear(in_p, out_p, bias=True)
        self.second_layer = nn.Linear(in_p, out_p, bias=True)

    def forward(self, x):
        first = self.first_layer(x)
        second = self.second_layer(x)
        return first + second