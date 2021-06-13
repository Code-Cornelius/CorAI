# normal libraries
import time
import pandas as pd
from tqdm import tqdm

# priv libraries
from priv_lib_plot import APlot
from priv_lib_util.tools.src.benchmarking import benchmark
from priv_lib_estimator.src.example_estimator.estim_benchmark_array import Plot_evol_benchmark_array, \
    Estim_benchmark_array


# section ######################################################################
#  #############################################################################
# helper functions


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

    for _ in range(number_of_reps):
        time = benchmark(func, number_of_rep=5, silent = True, *args, **kwargs)
        time_dict["Comput. Time"].append(time)

    estim.append(pd.DataFrame(time_dict))


# section ######################################################################
#  #############################################################################
# Benchmarking

# prepare data
import numpy as np

TEST = False
if TEST:
    powers = np.array(range(7, 12))
else:
    powers = np.array(range(6, 17))

number_of_reps = 1000
sizes = 2 ** powers

estim = Estim_benchmark_array()

for size in tqdm(sizes):
    test_arr = list(range(size))
    benchmark_and_save(estim, index_access, "index_access", arr=test_arr)
    benchmark_and_save(estim, elem_enum, "elem_enum", arr=test_arr)

# TODO NO INDEX
estim.df = estim.df.reset_index(drop = True)
time.sleep(1)
plot_evol_estim = Plot_evol_benchmark_array(estim)

plot_evol_estim.draw(feature_to_draw='Comput. Time', separator_colour='Method',
                     save_plot=False, dict_plot_for_main_line= {})
APlot.show_plot()