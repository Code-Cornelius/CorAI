import warnings
import matplotlib.pyplot as plt
import numpy as np

# my libraries:
from priv_lib_plot.src.acolor.acolorset import AColorset


class AColorsetContinuous(AColorset):
    def __init__(self, colors_name, nb_of_needed_colors=10, restrain_range_colors=(0, 1)):
        try:
            self.cm = plt.get_cmap(colors_name)
        except ValueError:
            warnings.warn("The ColorMap requested does not match to any existing cm. Use Jet instead.")
            self.cm = plt.get_cmap('jet')

        # cm are callable with the values at which we wish to get the colors:
        colors = self.cm(np.linspace(*restrain_range_colors, nb_of_needed_colors))
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

    cont_color = AColorsetContinuous('Reds', 100, (0.4, 1))

    plt.figure()
    for i, c in enumerate(cont_color):
        xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yy = np.sin(xx) + 2 * i
        plt.plot(xx, yy, color=c)

    plt.figure()
    for i in range(5):
        xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yy = np.sin(xx) + 2 * i
        plt.plot(xx, yy, color=cont_color[20 * i])

    plt.show()
