import torch.nn as nn

class Reshape(nn.Module):
    # reshape might be a view, but could also be a copy... since we are slicing in a non contiguous way.
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)