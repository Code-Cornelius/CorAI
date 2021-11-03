# normal libraries
from matplotlib import pyplot as plt  # plotting
import os

# priv_libraries
from corai_metaclass import Register, deco_register


class Displayable_plot(object, metaclass=Register):
    # _fig needs to be redefined lower class (APlot does it).

    _fig = None

    @deco_register
    def __init__(self):
        pass

    @staticmethod
    def show_and_continue(waiting_time=0.0001):
        plt.pause(waiting_time)
        return

    @staticmethod
    def show_plot():
        """
        Semantics:
            adapter for the show matplotlib.pyplot function.

        Returns:
            void.

        """
        plt.show()
        return

    def save_plot(self, name_save_file='plots/image'):
        """
        Semantics:
            saves the plot drawn.
            Create a directory if the path yields a non-existent directory.


        Args:
            name_save_file: path and name of the image. No extension.

        Returns:
            nothing.
        """
        directory_where_to_save = os.path.dirname(name_save_file)
        if not os.path.exists(directory_where_to_save):
            if directory_where_to_save != '':
                os.makedirs(directory_where_to_save)
        self._fig.savefig(name_save_file + '.png', dpi=500)
        return

    def tight_layout(self):
        self._fig.tight_layout()
        return
