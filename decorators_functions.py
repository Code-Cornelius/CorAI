import functools


class Memoization:
    def __init__(self, func, key_names):
        functools.update_wrapper(self, func)
        self.func = func
        self.key_names = key_names
        self.dictionary = {}

    def __call__(self, *args, **kwargs):
        keys = []

        # Todo: make safe, get might return none, explode if it does
        # get all the elements that build the key
        for key in self.key_names:
            keys.append(kwargs.get(key))

        # pack them into a tuple
        key = tuple(keys)

        if key not in self.dictionary:
            self.dictionary[key] = self.func(*args, **kwargs)

        return self.dictionary[key]

    def clear(self):
        self.dictionary.clear()
