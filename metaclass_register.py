# I want to create a metaclass which creates a dict in order to register the instances of a class.

import functools


def deco_register(func):
    # the idea is basically that wrapping this on the constructor, everytime I init, the object is added to the class list.
    @functools.wraps(func)
    def wrapper_register(*args, **kwargs):
        (args[0].__class__.list_register_instances).append(args[0])
        # __class__ is for class attribute.
        # in other words i wrote:         knowing that args[0] is self.
        # cls.list_register_instances  .append(self)
        return func(*args, **kwargs)

    return wrapper_register

# dict where I store the classes that appear in the registering.
dict_register_classes = {}

class register(type):
    # instructions :
        # before any class' init with that meta, put the decorator !

    # new is affecting the new classes created with that meta.
    def __new__(meta, name, bases, attrs):
        dict_register_classes[name] = cls = type.__new__(meta, name, bases, attrs)  # assigniation from right to left. cls is the new class created by new.

        cls.list_register_instances = []
        cls.print_register = classmethod(meta.print_register)  # a method inside metaclass is a class method.
        # On the other hand, classmethod is binding the method to both class method and object method scope.
        return cls

    def print_register(self):
        for element in self.list_register_instances:
            print(element)
        return self.list_register_instances

    # now I don't need that method anymore since I have classmethod on print_register.
    # def print_register_class(cls):
    #     for element in cls.list_register_instances:
    #         print(element)
    #     return cls.list_register_instances


#
# class Foo(metaclass=register):
#     @deco_register
#     def __init__(self):
#         print("inside : " , self.list_register_instances)
#         pass
#
#     def print_register(self):
#         print("coucou")
#
# class Boo(metaclass=register):
#     @deco_register
#     def __init__(self):
#         print("inside : " , self.list_register_instances)
#         pass
#     def print_register(self):
#         print("coucou")
#
# f = Foo()
# f = Foo()
# f_ = Foo()
# b = Boo()
# print(f.list_register_instances)
# print(b.list_register_instances)
# print(dict_register_classes)
#
# print("1")
# f.print_register()
# print("2")
# Foo.print_register_class()
# print("4")
# Foo.print_register()


# example without anything fancy:
class Foo:
    list_register_instances = []

    def __init__(self):
        self.__class__.list_register_instances.append(self)

    @classmethod
    def print_register(cls):
        for element in cls.list_register_instances:
            print(element)
