import warnings

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Collection, Iterable

import matplotlib.pyplot as plt
import numpy as np


class AColorset(Collection, metaclass=ABCMeta):
    """ Collection through self.colors. It means it is also an iterable.
    Wraps colormap from matplotlib in a convenient and easy way to manipulate.
    """

    def __init__(self, colors):
        """

        Args:
            colors: colors should be a collection (implements __contains__, __iter__, __len__)
        """
        self.colors = colors

    def __iter__(self):
        return iter(self.colors)

    def __contains__(self, item):
        return item in self.colors

    def __len__(self):
        return len(self.colors)

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
