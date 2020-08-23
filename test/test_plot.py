# normal libraries
import numpy as np  #maths library and arrays
import unittest

# my libraries

np.random.seed(124)

# errors:

# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# import numpy as np
# xx = np.linspace(0,1000,10000)
# yy = xx*2
# aplot = plot_functions.APlot(how = (1,1))
# aplot.uni_plot(nb_ax = 0, xx = xx, yy = yy)
# my_list = aplot.print_register()
#
# for i in my_list:
#     i.plot_vertical_line(200, np.linspace(0,10000,100000), nb_ax=0 )
# plt.show()


# xx = np.random.random(1000)
# aplot = plot_functions.APlot(how = (1,3))
# aplot.hist(xx, 0,
#            dict_param_hist= {"bins" : 60} )
# aplot.hist(xx, 1)
# aplot.hist(xx, 2)
# plt.show()




class Test_plot(unittest.TestCase):
    # section ######################################################################
    #  #############################################################################
    # setup

    def setUp(self):
        pass

    def tearDown(self):
        plt.show()

    # section ######################################################################
    #  #############################################################################
    # tests


    def test_1(self):
        pass










#test register
# import numpy as np
# xx = np.linspace(0,1000,10000)
# yy = xx*2
# aplot = APlot(how = (1,1))
# aplot.uni_plot(nb_ax = 0, xx = xx, yy = yy)
# my_list = aplot.print_register()
#
# for i in my_list:
#     i.plot_vertical_line(200, np.linspace(0,10000,100000), nb_ax=0 )
# plt.show()






# section ######################################################################
#  #############################################################################
# test


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
# class Foo:
#     list_register_instances = []
#
#     def __init__(self):
#         self.__class__.list_register_instances.append(self)
#
#     @classmethod
#     def print_register(cls):
#         for element in cls.list_register_instances:
#             print(element)
