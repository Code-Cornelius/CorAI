# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import scipy.integrate  # for quad
import cmath  # complex numbers
import time  # computational time

def function_vanilla_option_output(x, k):
    return max(0, x - np.exp(k))

