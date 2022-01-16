import torch
import torch.nn as nn


class One_hidden_recurrent(nn.Module):
    def __init__(self, num_layers, nb_directions, hidden_size):
        super().__init__()
        # h0:

        self.num_layers = num_layers
        self.nb_directions = nb_directions
        self.hidden_size = hidden_size

        self.hidden_state_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                       1,  # repeated later to have batch size
                                                       self.hidden_size),
                                           requires_grad=True)  # parameters are moved to device and learn.

    def get_hidden_states(self, batch_size):
        return self.hidden_state_0.repeat(batch_size)

    def forward(self, x):
        batch_size = (1, x.shape[0], 1)
        return (x, self.get_hidden_states(batch_size))
