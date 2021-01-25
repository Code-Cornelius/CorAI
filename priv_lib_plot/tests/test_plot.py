# normal libraries
import unittest
from PIL import ImageChops, Image  # image comparison

# other files
from priv_lib_plot import APlot



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def image_comparison(path1, path2):
    im1 = Image.open(path1).convert('RGB')  # we open the image with RGB code,
    # because some times there is noise on the image
    im2 = Image.open(path2).convert('RGB')  # same
    diff = ImageChops.difference(im2, im1)
    if diff.getbbox():  # if images are different
        return False
    else:
        return True


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
        pass
        # APlot.show_plot()

    def test_constructor_plot_data_directly(self):
        # direct plot only yy
        APlot(datay=yy)
        plt.savefig("image_reference_test_plot/test_image_yy.png")
        # direct plot xx and yy
        APlot(datax=xx, datay=yy)
        plt.savefig("image_reference_test_plot/test_image_xx_yy.png")
        assert (image_comparison("image_reference_test_plot/test_image_yy.png",
                                 "image_reference_test_plot/image_yy.png"))
        assert (image_comparison("image_reference_test_plot/test_image_xx_yy.png",
                                 "image_reference_test_plot/image_xx_yy.png"))

    def test_uniplot(self):
        # uni plot
        aplot_2 = APlot()
        aplot_2.uni_plot(0, xx, yy)
        plt.savefig("image_reference_test_plot/test_image_uniplot_1.png")

        # uni plot *4
        aplot_3 = APlot(how=(2, 2), sharex=True)
        aplot_3.uni_plot(0, xx, yy)
        aplot_3.uni_plot(1, xx, yy)
        aplot_3.uni_plot(2, xx, yy)
        aplot_3.uni_plot(3, xx, yy)
        plt.savefig("image_reference_test_plot/test_image_uniplot_2.png")

        assert (image_comparison("image_reference_test_plot/test_image_uniplot_1.png",
                                 "image_reference_test_plot/image_uniplot_1.png"))
        assert (image_comparison("image_reference_test_plot/test_image_uniplot_2.png",
                                 "image_reference_test_plot/image_uniplot_2.png"))

    def test_bi_plot(self):
        aplot_4 = APlot(how=(2, 2), sharey=True)
        aplot_4.bi_plot(0, 1, xx, yy, xx, yy)
        aplot_4.uni_plot(2, xx, yy)
        aplot_4.uni_plot(3, xx, yy)
        plt.savefig("image_reference_test_plot/test_image_biplot.png")
        assert (image_comparison("image_reference_test_plot/test_image_biplot.png",
                                 "image_reference_test_plot/image_biplot.png"))

    def test_plot_bis(self):
        # two plots same figures
        aplot_1 = APlot()
        aplot_1.uni_plot(0, xx, yy + 5)
        aplot_1.uni_plot_ax_bis(0, xx, np.exp(yy))
        plt.savefig("image_reference_test_plot/test_plot_bis.png")
        assert (image_comparison("image_reference_test_plot/test_plot_bis.png",
                                 "image_reference_test_plot/plot_bis.png"))

    def test_plot_function(self):
        def f(x):
            return 3 * x

        aplot = APlot()
        aplot.plot_function(f, xx)
        plt.savefig("image_reference_test_plot/test_plot_function.png")
        assert (image_comparison("image_reference_test_plot/test_plot_function.png",
                                 "image_reference_test_plot/plot_function.png"))

    def test_hist(self):
        aplot = APlot()
        aplot.hist(yy)
        plt.savefig("image_reference_test_plot/test_plot_hist.png")
        assert (image_comparison("image_reference_test_plot/test_plot_hist.png",
                                 "image_reference_test_plot/plot_hist.png"))


class Test_REGISTER(unittest.TestCase):
    def test_register(self):
        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=xx, yy=yy)
        my_list = aplot.print_register()

        for i in my_list:
            i.plot_vertical_line(200, np.linspace(0, 10000, 100000), nb_ax=0)
