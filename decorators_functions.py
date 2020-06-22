import functools
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
            #key = tuple(keys) # pack them into a tuple

            if key not in self.dictionary:
                self.dictionary[key] = self.func(*args, **kwargs)

            return self.dictionary[key]

        def clear(self):
            self.dictionary.clear()

    return MemoizationClass
