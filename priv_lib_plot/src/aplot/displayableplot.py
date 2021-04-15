# normal libraries
from matplotlib import pyplot as plt  # plotting

# my libraries
from priv_lib_metaclass import Register, deco_register


class Displayable_plot(object, metaclass=Register):

    @deco_register
    def __init__(self):
        pass

    @staticmethod
    def show_and_continue(waiting_time=0.0001):
        plt.pause(waiting_time)

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

    @staticmethod
    def save_plot(name_save_file='image'):
        """
        Semantics:
            saves the plot drawn.

        Args:
            name_save_file: name of the file. Can be used to chose the path.

        Returns:
            nothing.
        """
        plt.savefig(name_save_file + '.png', dpi=800)
        return
