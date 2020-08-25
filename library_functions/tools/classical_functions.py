# normal libraries


# my libraries

# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def curry(f):
    # curry is a concept transforming f(x,...,z) into f(x)...(z)

    argc = f.__code__.co_argcount
    f_args = []
    f_kwargs = {}

    def g(*args, **kwargs):
        nonlocal f_args, f_kwargs
        f_args += args
        f_kwargs.update(kwargs)
        if len(f_args) + len(f_kwargs) == argc:
            return f(*f_args, **f_kwargs)
        else:
            return g

    return g
