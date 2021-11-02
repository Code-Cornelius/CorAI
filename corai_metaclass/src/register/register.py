# I want to create a metaclass which creates a dict in order to register the instances of a class.
# dict where I store the classes that appear in the registering.
dict_register_classes = {}


class Register(type):
    # instructions:
    # before any class' init with that meta, put the decorator !

    # new is affecting the new classes created with that meta.
    def __new__(meta, name, bases, attrs):
        dict_register_classes[name] = cls = type.__new__(meta, name, bases, attrs)
        # assignation from right to left. cls is the new class created by new.

        cls.list_register_instances = []
        cls.get_register = classmethod(meta.get_register)  # a method inside metaclass is a class method.
        # On the other hand, class method is binding the method to both class method and object method scope.
        return cls

    def get_register(self, to_print = False):
        if to_print:
            for element in self.list_register_instances:
                print(element)
        return self.list_register_instances
