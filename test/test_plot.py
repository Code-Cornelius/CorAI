# normal libraries
import unittest

# other files
from library_classes.plot.class_aplot import *

np.random.seed(124)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Test_register(unittest.TestCase):
    def test_print_register(self):
        class Foo(metaclass=register):
            @deco_register
            def __init__(self):
                print("inside : ", self.list_register_instances)
                pass

            @staticmethod
            def print_register():
                print("hi")

        class Boo(metaclass=register):
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
        f.print_register()
        print("4")
        Foo.print_register()


xx = np.linspace(0, 10, 10000)
yy = np.cos(xx)


class Test_APLOT(unittest.TestCase):

    def tearDown(self):
        plt.show()

    def test_basic_plot(self):
        #direct plot only yy
        APlot(datay=yy)
        #direct plot xx and yy
        APlot(datax=xx, datay=yy)

        # uni plot
        aplot_2 = APlot()
        aplot_2.uni_plot(0, xx, yy)

        # uni plot *4
        aplot_3 = APlot(how=(2, 2), sharex=True)
        aplot_3.uni_plot(0, xx, yy)
        aplot_3.uni_plot(1, xx, yy)
        aplot_3.uni_plot(2, xx, yy)
        aplot_3.uni_plot(3, xx, yy)

        # uni plot good ordering of the plots ( TOP LEFT, TOP RIGHT, BOTTOM LEFT)
        aplot_4 = APlot(how=(2, 2), sharey=True)
        aplot_4.bi_plot(0, 1, xx, yy, xx, yy)
        aplot_4.uni_plot(2, xx, yy)
        aplot_4.uni_plot(3, xx, yy)


    def test_plot_bis(self):
        # two plots same figures
        aplot_1 = APlot()
        aplot_1.uni_plot_ax_bis(0, xx, yy + 5)
        aplot_1.uni_plot_ax_bis(0, xx, yy)

    def test_plot_function(self):
        def f(x):
            return 3 * x

        aplot = APlot()
        aplot.plot_function(f, xx)

    def test_hist(self):
        aplot = APlot()
        aplot.hist(yy)

    def test_register(self):
        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=xx, yy=yy)
        my_list = aplot.print_register()

        for i in my_list:
            i.plot_vertical_line(200, np.linspace(0, 10000, 100000), nb_ax=0)
