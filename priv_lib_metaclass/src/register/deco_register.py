import functools


def deco_register(func):
    # the idea is basically that wrapping this on the constructor,
    # everytime I init, the object is added to the class list.
    @functools.wraps(func)
    def wrapper_register(*args, **kwargs):
        args[0].__class__.list_register_instances.append(args[0])
        # __class__ is for class attribute.
        # in other words i wrote:         knowing that args[0] is self.
        # cls.list_register_instances  .append(self)
        return func(*args, **kwargs)

    return wrapper_register