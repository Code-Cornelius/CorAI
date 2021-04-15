import warnings

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


class AColorset(Sequence, metaclass=ABCMeta):
    """ Sequence through self.colors. It means it behaves like a list.
    Wraps colormap from matplotlib in a convenient and easy way to manipulate.
    """

    def __init__(self, colors):
        """

        Args:
            colors: colors should be a collection (implements __contains__, __iter__, __len__)
        """
        self.colors = colors

    # ITERABLE
    def __iter__(self):
        return iter(self.colors)

    # SIZED
    def __contains__(self, item):
        return item in self.colors

    # CONTAINER
    def __len__(self):
        return len(self.colors)

    # SEQUENCE
    def __getitem__(self, item):
        return self.colors[item]

    @property
    @abstractmethod
    def cm(self):
        return self._cm

    @cm.setter
    def cm(self, new_cm):
        """
        new_cm: should be a colormap from plt.cm
        """
        self._cm = new_cm
