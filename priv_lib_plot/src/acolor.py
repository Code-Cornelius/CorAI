import matplotlib.pyplot as plt
import numpy as np

color_DARK = plt.cm.Dark2.colors

nb_of_object = 10
color_RAINBOW = iter(plt.cm.rainbow(np.linspace(0, 1, len(nb_of_object))))
