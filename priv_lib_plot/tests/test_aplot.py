# normal libraries
import unittest
import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageChops, Image  # image comparison

# other files
from priv_lib_plot import APlot
from priv_lib_metaclass import Register,deco_register, dict_register_classes



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






class Test_APLOT(unittest.TestCase):

    def setUp(self) -> None:
        self.xx = np.linspace(0, 10, 10000)
        self.yy = np.cos(self.xx)
        return

    def tearDown(self):
        pass
        # APlot.show_plot()

    def test_constructor_plot_data_directly(self):
        # direct plot only yy
        APlot(datay=self.yy)
        plt.savefig("image_reference_test_plot/test_image_yy.png")
        # direct plot xx and yy
        APlot(datax=self.xx, datay=self.yy)
        plt.savefig("image_reference_test_plot/test_image_xx_yy.png")
        assert (image_comparison("image_reference_test_plot/test_image_yy.png",
                                 "image_reference_test_plot/image_yy.png"))
        assert (image_comparison("image_reference_test_plot/test_image_xx_yy.png",
                                 "image_reference_test_plot/image_xx_yy.png"))

    def test_uniplot(self):
        # uni plot
        aplot_2 = APlot()
        aplot_2.uni_plot(0, self.xx, self.yy)
        plt.savefig("image_reference_test_plot/test_image_uniplot_1.png")

        # uni plot *4
        aplot_3 = APlot(how=(2, 2), sharex=True)
        aplot_3.uni_plot(0, self.xx, self.yy)
        aplot_3.uni_plot(1, self.xx, self.yy)
        aplot_3.uni_plot(2, self.xx, self.yy)
        aplot_3.uni_plot(3, self.xx, self.yy)
        plt.savefig("image_reference_test_plot/test_image_uniplot_2.png")

        assert (image_comparison("image_reference_test_plot/test_image_uniplot_1.png",
                                 "image_reference_test_plot/image_uniplot_1.png"))
        assert (image_comparison("image_reference_test_plot/test_image_uniplot_2.png",
                                 "image_reference_test_plot/image_uniplot_2.png"))

    def test_bi_plot(self):
        aplot_4 = APlot(how=(2, 2), sharey=True)
        aplot_4.bi_plot(0, 1, self.xx, self.yy, self.xx, self.yy)
        aplot_4.uni_plot(2, self.xx, self.yy)
        aplot_4.uni_plot(3, self.xx, self.yy)
        plt.savefig("image_reference_test_plot/test_image_biplot.png")
        assert image_comparison("image_reference_test_plot/test_image_biplot.png",
                                 "image_reference_test_plot/image_biplot.png")

    def test_plot_bis(self):
        # two plots same figures
        aplot_1 = APlot()
        aplot_1.uni_plot(0, self.xx, self.yy + 5)
        aplot_1.uni_plot_ax_bis(0, self.xx, np.exp(self.yy))
        plt.savefig("image_reference_test_plot/test_plot_bis.png")
        assert image_comparison("image_reference_test_plot/test_plot_bis.png",
                                 "image_reference_test_plot/plot_bis.png")

    def test_plot_function(self):
        def f(x):
            return 3 * x

        aplot = APlot()
        aplot.plot_function(f, self.xx)
        plt.savefig("image_reference_test_plot/test_plot_function.png")
        assert image_comparison("image_reference_test_plot/test_plot_function.png",
                                 "image_reference_test_plot/plot_function.png")

    def test_hist(self):
        aplot = APlot()
        aplot.hist(self.yy)
        plt.savefig("image_reference_test_plot/test_plot_hist.png")
        assert image_comparison("image_reference_test_plot/test_plot_hist.png",
                                 "image_reference_test_plot/plot_hist.png")


