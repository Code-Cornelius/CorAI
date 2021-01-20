import functools
import time

import priv_lib_util.tools


def Memoization(key_names):
    class MemoizationClass:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func # the function using the keys.
            self.dictionary = {} # the initial dict with the memoization data
            self.key_names = key_names # the keys that will be used for memoization

        def __call__(self, *args, **kwargs):
            keys = []
            # retrieve from the call arguments the keys corresponding to key_names.
            for key in self.key_names:
                keys.append(kwargs[key])
            key = (*keys,)
            # key = tuple(keys) # pack them into a tuple

            if key not in self.dictionary:
                self.dictionary[key] = self.func(*args, **kwargs)

            return self.dictionary[key]

        def clear(self):
            self.dictionary.clear()

    return MemoizationClass


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # time.perf_counter() the most precise available clock.
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def set_new_methods(**kwargs):
    """ set a set of new methods to a class, any quantity of methods.

    Args:
        **kwargs: name of method given by key, body by value.

    Returns:
        It returns the new class
    """

    def wrapper(cls):
        for key in kwargs:
            setattr(cls, key, kwargs[key])
        return cls

    return wrapper


def prediction_total_time(total_nb_tries, multiplicator_factor, actual_state):
    """

    Args:
        total_nb_tries: total number of iteration, this is the complexity of the algo.
        multiplicator_factor: how the complexity of the function evolves throughout the loop.
        actual_state: how much to reduce the time incrementally. I don't know how to do properly. I need an object that changes inside the function but from outside.
        for that reason I use the property of mutable object. ACTUAL_STATE IS A LIST!

    Returns:

    """

    def decorator_prediction_total_time(func):
        list_deco_estimation_times = []
        beginning_time = time.perf_counter()

        @functools.wraps(func)
        def wrapper_estimation_timer(*args, **kwargs):
            start_time = time.perf_counter()  # time.perf_counter() the most precise available clock.
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            list_deco_estimation_times.append(run_time)
            total_run_time = priv_lib_util.tools.function_iterable.mean_list(
                list_deco_estimation_times) * (total_nb_tries - actual_state[0]) * multiplicator_factor
            s, m, h, _ = priv_lib_util.tools.benchmarking.time_convertor(total_run_time,
                                                                         time_format=2)  # the _ is second frac.
            ts, tm, th = priv_lib_util.tools.benchmarking.time_text(s, m, h, 0)
            str1 = ''.join([th, tm, ts])

            total_run_time = time.perf_counter() - beginning_time
            s, m, h, _ = priv_lib_util.tools.benchmarking.time_convertor(total_run_time,
                                                                         time_format=2)  # the _ is second frac.
            ts, tm, th = priv_lib_util.tools.benchmarking.time_text(s, m, h, 0)
            str2 = ''.join([th, tm, ts])

            print(''.join(["/" * 15, f"estimated time left before completion: {str1}. Total time: {str2}.", "/" * 15]))
            # TODO 20/07/2020 nie_k: perhaps print iff the actual state is in a certain position
            return value

        return wrapper_estimation_timer

    return decorator_prediction_total_time

# test
# import numpy as np
#
# N1 = 10
# N2 = 10
# total_nb_tries = N1 * N2
# actual_state = [0]
#
#
#
#
#
#
# @prediction_total_time(total_nb_tries = total_nb_tries,
#                        multiplicator_factor = 1,
#                        actual_state = actual_state)
# def f():
#     A = np.full((10000,1000),10)
#     np.exp(A)
#
#
#
# for j in range(N1):
#     for i in range(N2):
#         actual_state[0] += 1
#         f()
