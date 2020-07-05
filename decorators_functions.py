import functools
import time
from inspect import signature  # know signature of a function


def Memoization(key_names):
    class MemoizationClass:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func
            self.dictionary = {}
            self.key_names = key_names
            # No default behaviour
            # if key_names is not None:
            #     self.key_names = key_names
            # else:
            #     self.key_names = {(signature(self.func).parameters.values())[0]: 0}  # getting the first parameter

        def __call__(self, *args, **kwargs):
            keys = []
            # get all the elements that build the key
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
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def set_new_methods(**kwargs):
    ''' set new methods to a class, any number.

    Args:
        **kwargs: name of method given by key, body by value.

    Returns:

    '''

    def wrapper(cls):
        for key in kwargs:
            setattr(cls, key, kwargs[key])
        return cls

    return wrapper


def set_new_class_methods(**kwargs):
    ''' set new methods to a class, any number.

    Args:
        **kwargs: name of method given by key, body by value.

    Returns:

    '''

    def wrapper(cls):
        for key in kwargs:
            setattr(cls, key, classmethod(kwargs[key]))
        return cls

    return wrapper
