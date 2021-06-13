# normal libraries
import time
import pandas as pd

# priv libraries
from priv_lib_util.tools.src.benchmarking import benchmark

# section ######################################################################
#  #############################################################################
# helper functions
from priv_lib_estimator.src.example_estimator.estim_benchmark_array import Plot_evol_benchmark_array, \
    Estim_benchmark_array


def index_access(arr):
    for i in range(len(arr)):
        arr[i] + 1
        i + 1


def elem_enum(arr):
    for i, elem in enumerate(arr):
        elem + 1
        i + 1


def benchmark_and_save(estim, func, method_name, number_of_reps=100, *args, **kwargs):
    time_dict = {"Method": [method_name] * number_of_reps,
                 "Array Size": [len(test_arr)] * number_of_reps,
                 "Comput. Time": []}

    for i in range(number_of_reps):
        time = benchmark(func, number_of_rep=1, *args, **kwargs)
        time_dict["Comput. Time"].append(time)

    estim.append(pd.DataFrame(time_dict))


# section ######################################################################
#  #############################################################################
# Benchmarking

# prepare data
import numpy as np

TEST = True
if TEST:
    powers = np.array(range(7, 12))
else:
    powers = np.array(range(7, 20))

number_of_reps = 100
sizes = 2 ** powers

estim = Estim_benchmark_array()

for size in sizes:
    test_arr = list(range(size))
    benchmark_and_save(estim, index_access, "index_access", arr=test_arr)
    benchmark_and_save(estim, elem_enum, "elem_enum", arr=test_arr)

time.sleep(1)
estim_hist = Plot_evol_benchmark_array(estim)

estim_hist.draw(feature_to_draw='Comput. Time', separator_colour='Method')
