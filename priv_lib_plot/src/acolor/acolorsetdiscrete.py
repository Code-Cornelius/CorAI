import warnings
import matplotlib.pyplot as plt
import numpy as np

# priv_libraries:
from priv_lib_plot.src.acolor.acolorset import AColorset


class AColorsetDiscrete(AColorset):
    """
    """

    def __init__(self, colors_name):
        try:
            self.cm = plt.get_cmap(colors_name)
        except ValueError:
            warnings.warn("The ColorMap requested does not match to any existing cm. Use Dark2 instead.")
            self.cm = plt.get_cmap('Dark2')

        # cm are callable with the values at which we wish to get the colors:
        colors = self.cm.colors
        super().__init__(colors)

    @property
    def cm(self):
        return self._cm

    @cm.setter
    def cm(self, new_cm):
        """
        new_cm: should be a colormap from plt.cm
        """
        self._cm = new_cm


if __name__ == '__main__':

    disc_color = AColorsetDiscrete('Dark2')

    plt.figure()
    for i, c in enumerate(disc_color):
        xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yy = np.sin(xx) + 2 * i
        plt.plot(xx, yy, color=c)

    plt.show()
