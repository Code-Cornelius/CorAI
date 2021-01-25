# normal libraries
import unittest
import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageChops, Image  # image comparison

# other files
from priv_lib_plot import APlot
from priv_lib_metaclass import Register,deco_register, dict_register_classes



#TODO MAKE IT PROPERLY
class Test_register(unittest.TestCase):
    def test_print_register(self):
        class Foo(metaclass=Register):
            @deco_register
            def __init__(self):
                print("inside : ", self.list_register_instances)
                pass

            @staticmethod
            def print_register():
                print("hi")

        class Boo(metaclass=Register):
            @deco_register
            def __init__(self):
                print("inside : ", self.list_register_instances)
                pass

            @staticmethod
            def print_register():
                print("hi")

        f = Foo()
        f = Foo()
        f_ = Foo()
        b = Boo()
        print(f.list_register_instances)
        print(b.list_register_instances)

        print(dict_register_classes)

        print("1")
        f.get_register()
        print("4")
        Foo.get_register()




