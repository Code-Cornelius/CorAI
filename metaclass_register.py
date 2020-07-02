# I want to create a metaclass which creates a dict in order to register the instances of a class.
import functools


def dec_register(func):
    @functools.wraps(func)
    def wrapper_register(*args, **kwargs):
        (args[0].__class__.list_register_instances).append(args[0])
        return func(*args, **kwargs)
    return wrapper_register




dict_register_classes = {}
class register(type):
    def __new__(meta, name, bases, attrs):
        dict_register_classes[name] = cls = type.__new__(meta, name, bases, attrs)  # assigniation from right to left
        cls.list_register_instances = []
        cls.print_register = meta.print_register
        return cls

    def print_register(self):
        for element in self.list_register_instances:
            print(element)



#
class Foo(metaclass=register):
    @dec_register
    def __init__(self):
        pass

    def print_register(self):
        print("coucou")

class Boo(metaclass=register):
    @dec_register
    def __init__(self):
        pass
    def print_register(self):
        print("coucou")

f = Foo()
f_ = Foo()
b = Boo()
print(f.list_register_instances)
print(b.list_register_instances)

print(dict_register_classes)

f.print_register()



# example without anything fancy:
class Foo:
    instances = []
    def __init__(self):
        self.__class__.instances.append(self)

    @classmethod
    def printInstances(cls):
        for instance in cls.instances:
            print(instance)
Foo.printInstances()