from priv_lib_estimator.src.example_sub_estimator_class.estim_time_size import Estim_time_size
from priv_lib_util.tools.src.benchmarking import benchmark
import pandas as pd

class Estim_benchmark_array(Estim_time_size):
    NAMES_COLUMNS = Estim_time_size.NAMES_COLUMNS.copy()
    NAMES_COLUMNS.add("Method")

    def __init__(self):
        super().__init__()




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
    time_dict = {
        "Method": [],
        "N": [],
        "TIME": []
    }

    time_dict["Method"] = [method_name] * number_of_reps
    time_dict["N"] = [len(test_arr)] * number_of_reps

    for i in range(number_of_reps):
        time = benchmark(func, number_of_rep=1, *args, **kwargs)
        time_dict["TIME"].append(time)

    estim.append(pd.DataFrame(time_dict))

# section ######################################################################
#  #############################################################################
# Benchmarking

# prepare data
import numpy as np
powers = np.array(range(10,20))

number_of_reps = 100
sizes = 2 ** powers

estim = Estim_benchmark_array()

for size in sizes:
    test_arr = list(range(size))
    benchmark_and_save(estim, index_access, "index_access", arr=test_arr)
    benchmark_and_save(estim, elem_enum, "elem_enum", arr=test_arr)
    # range_size = increment_factor * range_size

print(estim)