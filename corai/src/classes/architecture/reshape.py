import torch.nn as nn

class Reshape(nn.Module):
    """
    References:
        https://pytorch.org/docs/master/generated/torch.reshape.html#torch.reshape
            turns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.
        https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)