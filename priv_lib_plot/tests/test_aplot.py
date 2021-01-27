# normal libraries
import unittest
from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageChops, Image  # image comparison

# other files
from priv_lib_plot import APlot
from priv_lib_metaclass import Register, deco_register, dict_register_classes


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


class Test_APlot(TestCase):
    def setUp(self) -> None:
        self.xx = np.linspace(0, 10, 10000)
        self.yy = np.abs(np.cos(self.xx)) + 1
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



    def test_set_dict_ax_and_bis(self):
        # first, trying every simple possibility for dict_ax
        with self.subTest('title'):
            aplot = APlot()
            dict_plot = {'title': 'my title'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_title.png")

        with self.subTest('xlabel'):
            aplot = APlot()

            dict_plot = {'xlabel': 'my x label'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xlabel.png")

        with self.subTest('ylabel'):
            aplot = APlot()

            dict_plot = {'ylabel': 'my y label'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_ylabel.png")

        with self.subTest('xscale'):
            aplot = APlot()

            dict_plot = {'xscale': 'log'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xscale.png")

        with self.subTest('yscale'):
            aplot = APlot()

            dict_plot = {'yscale': 'log'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_yscale.png")

        with self.subTest('xint'):
            aplot = APlot()

            dict_plot = {'xint': True}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xint.png")

        with self.subTest('xint_yint'):
            aplot = APlot()

            dict_plot = {'xint': True, 'yint': True}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xyint.png")

        with self.subTest('parameter'):
            aplot = APlot()

            dict_plot = {
                'parameters': ['A'], 'name_parameters': ['A']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_parameter.png")

        with self.subTest('parameters'):
            aplot = APlot()

            dict_plot = {
                'parameters': ['A', 3, 5, 10, 42], 'name_parameters': ['A', 'B', '$\sigma$', '$\\rho$', 'C']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_parameters.png")

        with self.subTest('parameters not good length'):
            aplot = APlot()

            dict_plot = {
                'parameters': ['A', 3, 5, 10, 42], 'name_parameters': ['A']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

        with self.subTest('xlim'):
            aplot = APlot()

            dict_plot = {'xlim': [0, 0.5]}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xlim.png")

        with self.subTest('ylim'):
            aplot = APlot()

            dict_plot = {'ylim': [1, 5]}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_ylim.png")

        # second, try the same as before where the second axis is also setting another characteristic.
        with self.subTest('title2'):
            aplot = APlot()
            dict_plot1 = {'title': 'my title1'}
            dict_plot2 = {'title': 'my title2'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_title_2.png")

        with self.subTest('xlabel2'):
            aplot = APlot()

            dict_plot1 = {'xlabel': 'my x label1'}
            dict_plot2 = {'xlabel': 'my x label2'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xlabel_2.png")

        with self.subTest('ylabel2'):
            aplot = APlot()

            dict_plot1 = {'ylabel': 'my y label1'}
            dict_plot2 = {'ylabel': 'my y label2'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_ylabel_2.png")










        with self.subTest('xscale'):
            aplot = APlot()

            dict_plot = {'xscale': 'log'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xscale.png")

        with self.subTest('yscale'):
            aplot = APlot()

            dict_plot = {'yscale': 'log'}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_yscale.png")

        with self.subTest('xint'):
            aplot = APlot()

            dict_plot = {'xint': True}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xint.png")

        with self.subTest('xint_yint'):
            aplot = APlot()

            dict_plot = {                        'xint': True, 'yint': True}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xyint.png")



        with self.subTest('parameter'):
            aplot = APlot()

            dict_plot = {
                         'parameters': ['A'], 'name_parameters': ['A']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_parameter.png")

        with self.subTest('parameters'):
            aplot = APlot()

            dict_plot = {
                'parameters': ['A',3,5,10,42], 'name_parameters': ['A','B','$\sigma$','$\\rho$', 'C']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_parameters.png")

        with self.subTest('parameters not good length'):
            aplot = APlot()

            dict_plot = {
                'parameters': ['A', 3, 5, 10, 42], 'name_parameters': ['A']}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)





        with self.subTest('xlim'):
            aplot = APlot()

            dict_plot = {'xlim': [0,0.5]}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xlim.png")

        with self.subTest('ylim'):
            aplot = APlot()

            dict_plot = {'ylim': [1,5]}

            aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            plt.savefig("image_reference_test_plot/test_image_set_dict_ax_ylim.png")






















        aplot = APlot()

        dict_plot = {'title': 'my title',
                     'xlabel': 'my x label', 'ylabel': 'my ylabel',
                     'xscale': 'log', 'yscale': 'log',
                     'basex': 10, 'basey': 10,
                     'xint': False, 'yint': False,
                     'parameters': None, 'name_parameters': None,
                     'xlim': None, 'ylim': None}

        aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
        aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

        plt.savefig("image_reference_test_plot/test_image_set_dict_ax_same_dict.png")

        aplot = APlot()
        dict_plot = {'title': 'my title1',
                     'xlabel': 'my x label1', 'ylabel': 'my ylabel1',
                     'xscale': 'log', 'yscale': 'log',
                     'basex': 2, 'basey': 2,
                     'xint': False, 'yint': False,
                     'parameters': None, 'name_parameters': None,
                     'xlim': None, 'ylim': None}

        dict_plot = {'title': 'my title2',
                     'xlabel': 'my x label2', 'ylabel': 'my ylabel2',
                     'xscale': 'log', 'yscale': 'log',
                     'basex': 10, 'basey': 10,
                     'xint': False, 'yint': False,
                     'parameters': None, 'name_parameters': None,
                     'xlim': None, 'ylim': None}

        aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
        aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)

        plt.savefig("image_reference_test_plot/test_image_set_dict_ax_different_dict.png")

        aplot = APlot()

        dict_plot = {'title': 'my title',
                     'xlabel': 'my x label', 'ylabel': 'my ylabel',
                     'xscale': 'log', 'yscale': 'log',
                     'basex': 10, 'basey': 10,
                     'xint': True, 'yint': True,
                     'parameters': [3], 'name_parameters': ["$\sigma$"] ,
                     'xlim': [0,1], 'ylim': [1,5]}

        aplot.uni_plot(0, self.xx, self.yy, dict_ax=dict_plot)
        aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)

        plt.savefig("image_reference_test_plot/test_image_set_dict_ax_dict2.png")

    def test_show_legend(self):
        pass

    def test_tight_layout(self):
        pass

    def test_show_plot(self):
        pass

    def test_save_plot(self):
        pass
    #todo
    def test_uni_plot(self):
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

    # todo
    def test_bi_plot(self):
        aplot_4 = APlot(how=(2, 2), sharey=True)
        aplot_4.bi_plot(0, 1, self.xx, self.yy, self.xx, self.yy)
        aplot_4.uni_plot(2, self.xx, self.yy)
        aplot_4.uni_plot(3, self.xx, self.yy)
        plt.savefig("image_reference_test_plot/test_image_biplot.png")
        assert image_comparison("image_reference_test_plot/test_image_biplot.png",
                                "image_reference_test_plot/image_biplot.png")


    # todo
    def test_uni_plot_ax_bis(self):
        # two plots same figures
        aplot_1 = APlot()
        aplot_1.uni_plot(0, self.xx, self.yy + 5)
        aplot_1.uni_plot_ax_bis(0, self.xx, np.exp(self.yy))
        plt.savefig("image_reference_test_plot/test_plot_bis.png")
        assert image_comparison("image_reference_test_plot/test_plot_bis.png",
                                "image_reference_test_plot/plot_bis.png")

    # todo
    def test_cumulative_plot(self):
        self.fail()

    # todo
    def test_hist(self):
        aplot = APlot()
        aplot.hist(self.yy)
        plt.savefig("image_reference_test_plot/test_plot_hist.png")
        assert image_comparison("image_reference_test_plot/test_plot_hist.png",
                                "image_reference_test_plot/plot_hist.png")

    # todo
    def test_plot_function(self):
        def f(x):
            return 3 * x

        aplot = APlot()
        aplot.plot_function(f, self.xx)
        plt.savefig("image_reference_test_plot/test_plot_function.png")
        assert image_comparison("image_reference_test_plot/test_plot_function.png",
                                "image_reference_test_plot/plot_function.png")

    # todo
    def test_plot_vertical_line(self):
        self.fail()

    # todo
    def test_plot_line(self):
        self.fail()

    # todo
    def test_plot_point(self):
        self.fail()

    def test_help_dict_plot(self):
        pass

    def test_help_dict_ax(self):
        pass

