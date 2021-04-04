import functools


def deco_register(func):
    # the idea is basically that we should wrap every class' init function that uses register as metaclass.
    # we do this in order to append to the list of list_register_instances the object just created.
    # everytime I init, the object is added to the class list.
    @functools.wraps(func)
    def wrapper_register(*args, **kwargs):
        args[0].__class__.list_register_instances.append(args[0])
        # __class__ is for class attribute.
        # knowing that args[0] is self.
        # in other words i wrote:
        # cls.list_register_instances.append(self)
        # for list_register_instances go see register metaclass.
        return func(*args, **kwargs)

    return wrapper_register
