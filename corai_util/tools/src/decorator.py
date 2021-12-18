import functools
import time

import corai_util.tools
import signal
import logging


def Memoization(key_names):
    class MemoizationClass:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func  # the function using the keys.
            self.dictionary = {}  # the initial dict with the memoization data
            self.key_names = key_names  # the keys that will be used for memoization

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


def set_new_methods(**kwargs):
    """
    Semantics:
        Set a set of new methods to a class, any quantity of methods.

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


def estimation_remaining_time_computation(total_nb_tries, multiplicator_factor, actual_state):
    """
    Semantics:
        Decorator for a function that predicts the amount of time left to do a task.
        In order to do so, it compares the actual state of processing with respect to the total amount.
        tqdm might be easier to use, but does not provide a multiplicator factor.

    Args:
        total_nb_tries: total number of iterations, this is the total complexity.
        multiplicator_factor: how the complexity of the function evolves throughout the loop.
        actual_state: how much to reduce the time incrementally. I don't know how to do properly.
            I need an object that changes inside the function but from outside.
            for that reason I use the property of mutable object. ACTUAL_STATE IS A LIST!

    """

    def decorator_prediction_total_time(func):
        list_deco_estimation_times = []
        beginning_time = time.perf_counter()

        @functools.wraps(func)
        def wrapper_estimation_timer(*args, **kwargs):
            # estimation:
            start_time = time.perf_counter()  # time.perf_counter() the most precise available clock.
            value = func(*args, **kwargs)
            end_time = time.perf_counter()

            # computations and saving / printing
            run_time = end_time - start_time
            list_deco_estimation_times.append(run_time)
            total_run_time = corai_util.tools.function_iterable.mean_list(list_deco_estimation_times) *\
                             (total_nb_tries - actual_state[0]) * \
                             multiplicator_factor
            s, m, h, _ = corai_util.tools.benchmarking.time_convertor_sec2hours_min_sec(total_run_time,
                                                                                           time_format=2)  # the _ is second frac.
            ts, tm, th = corai_util.tools.benchmarking.time_time2text(s, m, h, 0)
            str1 = ''.join([th, tm, ts])

            total_run_time = time.perf_counter() - beginning_time
            s, m, h, _ = corai_util.tools.benchmarking.time_convertor_sec2hours_min_sec(total_run_time,
                                                                                           time_format=2)  # the _ is second frac.
            ts, tm, th = corai_util.tools.benchmarking.time_time2text(s, m, h, 0)
            str2 = ''.join([th, tm, ts])

            print(''.join(["/" * 15, f"estimated time left before completion: {str1}. Total time: {str2}.", "/" * 15]))
            # TODO 20/07/2020 nie_k: perhaps print iff the actual state is in a certain position
            return value

        return wrapper_estimation_timer

    return decorator_prediction_total_time


class DelayedKeyboardInterrupt:
    """
    Semantics:
        Class to be used to delay a keyboard interrupt for a critical section.

    How to use:
    with DelayedKeyboardInterrupt():
        # critical section
    """

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('Cannot interrupt while handling files. '
                      'The interrupt will be delayed until file operations are finished')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
