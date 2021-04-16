# normal libraries
import unittest
from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageChops, Image  # image comparison

import os
import pathlib
# other files
from priv_lib_plot import APlot
from config import ROOT_DIR
from priv_lib_metaclass import Register, deco_register, dict_register_classes

"""
    Auto
    False - manual testing 
          - save the file and check manually if the behaviour is as expected
    True - automatic testing
         - save the file with test_name and compare against manually approved correct image 
    ! Only use false locally    
"""

AUTO = True
DELETE_TEST_PLOT = True
PATH = os.path.join(ROOT_DIR, 'priv_lib_plot', 'tests', 'image_reference_test_plot')


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
        self.yy = np.cos(self.xx)
        self.image_name = ""
        return

    def tearDown(self):
        if DELETE_TEST_PLOT:
            for file in os.listdir(PATH):
                if file.startswith("test") and file.endswith(".png"):
                    file_path = os.path.join(PATH, file)
                    os.remove(file_path)

    def check_plot(self):
        image_path = os.path.join(PATH, f"{self.image_name}.png")
        if AUTO:
            test_path = os.path.join(PATH, f"test_{self.image_name}.png")
            plt.savefig(test_path, dpi=50)
            assert (image_comparison(test_path, image_path))
        else:
            plt.savefig(image_path, dpi=50)

    def test_constructor_plot_data_directly_only_yy(self):
        # direct plot only yy
        APlot(datay=self.yy)
        self.image_name = "image_yy"
        self.check_plot()

    def test_constructor_plot_data_directly_xx_yy(self):
        # direct plot xx and yy
        APlot(datax=self.xx, datay=self.yy)
        self.image_name = "image_xx_yy"
        self.check_plot()


    def test_set_dict_ax_and_bis_each_parameter_only_first_two_axis_same_graph(self):
        # first, trying every simple possibility for dict_ax
        yy = np.abs(np.cos(self.xx)) + 1
        with self.subTest('title'):
            aplot = APlot()
            dict_plot = {'title': 'my title'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_title"
            self.check_plot()

        with self.subTest('xlabel'):
            aplot = APlot()

            dict_plot = {'xlabel': 'my x label'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_xlabel"
            self.check_plot()

        with self.subTest('ylabel'):
            aplot = APlot()

            dict_plot = {'ylabel': 'my y label'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_ylabel"
            self.check_plot()

        with self.subTest('xscale'):
            aplot = APlot()

            dict_plot = {'xscale': 'log'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)
            self.image_name = "image_set_dict_ax_xscale_only_principal_axis"
            self.check_plot()

        with self.subTest('yscale left'):
            aplot = APlot()

            dict_plot = {'yscale': 'log'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_yscale"
            self.check_plot()

        with self.subTest('xint'):
            aplot = APlot()

            dict_plot = {'xint': True}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 1/3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_xint"
            self.check_plot()

        with self.subTest('xint_yint'):
            aplot = APlot()

            dict_plot = {'xint': True, 'yint': True}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 1/3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_xyint"
            self.check_plot()

        for i in range(11):
            with self.subTest('parameter', i=i):
                aplot = APlot()

                dict_plot = {'parameters': ['A', 3, 5]*i, 'name_parameters': ['A', '$\sigma$', '$\\rho$']*i}

                aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
                aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

                self.image_name = f"image_set_dict_ax_parameter_{i}"
                self.check_plot()


        # with self.subTest('parameters not good length'):
        #     aplot = APlot()
        #
        #     dict_plot = {
        #         'parameters': ['A', 3, 5, 10, 42], 'name_parameters': ['A']}
        #
        #     aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
        #     aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

        with self.subTest('xlim'):
            aplot = APlot()

            dict_plot = {'xlim': [0, 0.5]}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_xlim"
            self.check_plot()

        with self.subTest('ylim'):
            aplot = APlot()

            dict_plot = {'ylim': [1.2, 2.4]}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1)

            self.image_name = "image_set_dict_ax_ylim"
            self.check_plot()

    def test_set_dict_ax_and_bis_each_parameter_both_two_axis_same_graph(self):
        yy = np.abs(np.cos(self.xx)) + 1
        # second, try the same as before where the second axis is also setting another characteristic.
        with self.subTest('two different titles'):
            aplot = APlot()
            dict_plot1 = {'title': 'my title1'}
            dict_plot2 = {'title': 'my title2'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_title_2"
            self.check_plot()

        with self.subTest('two xlabels'):
            aplot = APlot()

            dict_plot1 = {'xlabel': 'my x label1'}
            dict_plot2 = {'xlabel': 'my x label2'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_xlabel_2"
            self.check_plot()

        with self.subTest('two ylabels'):
            aplot = APlot()

            dict_plot1 = {'ylabel': 'my y label1'}
            dict_plot2 = {'ylabel': 'my y label2'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_ylabel_2"
            self.check_plot()

        with self.subTest('xscale same both axis'):
            aplot = APlot()

            dict_plot = {'xscale': 'log'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)

            self.image_name = "image_set_dict_ax_xscale_both_same_scale"
            self.check_plot()

        with self.subTest('yscale same both axis'):
            aplot = APlot()

            dict_plot = {'yscale': 'log'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)

            self.image_name = "image_set_dict_ax_yscale_2_same"
            self.check_plot()

        with self.subTest('xscale different'):
            aplot = APlot()

            dict_plot2 = {'xscale': 'log'}
            dict_plot = {'xscale': 'linear'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_xscale_2_different_scales"
            self.check_plot()

        with self.subTest('yscale different'):
            aplot = APlot()

            dict_plot = {'yscale': 'log'}
            dict_plot2 = {'yscale': 'linear'}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_yscale_2_different"
            self.check_plot()

        # with self.subTest('xint same'):
        #     aplot = APlot()
        #
        #     dict_plot = {'xint': True}
        #
        #     aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
        #     aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)
        #
        #     plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xint_2_same.png")
        #
        # with self.subTest('xint_yint same'):
        #     aplot = APlot()
        #
        #     dict_plot = {'xint': True, 'yint': True}
        #
        #     aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
        #     aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot)
        #
        #     plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xyint_same_2.png")

        # with self.subTest('xint different'):
        #     aplot = APlot()
        #
        #     dict_plot = {'xint': True}
        #     dict_plot2 = {'xint': False}
        #
        #     aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
        #     aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)
        #
        #     plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xint_2_different_2.png")
        #
        # with self.subTest('xint_yint different'):
        #     aplot = APlot()
        #
        #     dict_plot = {'xint': True, 'yint': True}
        #     dict_plot2 = {'xint': True, 'yint': False}
        #
        #     aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot)
        #     aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)
        #
        #     plt.savefig("image_reference_test_plot/test_image_set_dict_ax_xyint_different_2.png")

        with self.subTest('xlim'):
            aplot = APlot()

            dict_plot1 = {'xlim': [0, 0.5]}
            dict_plot2 = {'xlim': [0, 1.5]}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_xlim_2"
            self.check_plot()

        with self.subTest('ylim'):
            aplot = APlot()

            dict_plot1 = {'ylim': [1, 5]}
            dict_plot2 = {'ylim': [1, 15]}

            aplot.uni_plot(0, self.xx, yy, dict_ax=dict_plot1)
            aplot.uni_plot_ax_bis(0, self.xx, 3 * np.sin(self.xx) + 1, dict_ax=dict_plot2)

            self.image_name = "image_set_dict_ax_ylim_2"
            self.check_plot()


    def test_show_legend(self):
        pass

    def test_tight_layout(self):
        pass


    def test_uni_plot_one_plot(self):
        # uni plot
        aplot_2 = APlot()
        aplot_2.uni_plot(0, self.xx, self.yy)
        aplot_2.tight_layout()

        self.image_name = "image_uniplot_1"
        self.check_plot()

    def test_uni_plot_for_plots_same_graph(self):
        # uni plot *4
        aplot_3 = APlot(how=(2, 2))
        aplot_3.uni_plot(0, self.xx, self.yy)
        aplot_3.uni_plot(1, self.xx, self.yy)
        aplot_3.uni_plot(2, self.xx, self.yy)
        aplot_3.uni_plot(3, self.xx, self.yy)
        aplot_3.tight_layout()

        self.image_name = "image_uniplot_2"
        self.check_plot()

    def test_uni_plot_for_plots_same_graph_sharex(self):
        # uni plot *4
        aplot_3 = APlot(how=(2, 2), sharex=True)
        aplot_3.uni_plot(0, self.xx, self.yy)
        aplot_3.uni_plot(1, self.xx, self.yy)
        aplot_3.uni_plot(2, self.xx, self.yy)
        aplot_3.uni_plot(3, self.xx, self.yy)
        aplot_3.tight_layout()

        self.image_name = "image_uniplot_2_sharex"
        self.check_plot()

    def test_uni_plot_for_plots_same_graph_sharex_not_same_interval_for_x(self):
        # uni plot *4
        xx = np.linspace(-1, 1, 1000)
        aplot_3 = APlot(how=(2, 2), sharex=True)
        aplot_3.uni_plot(0, xx, np.exp(xx))
        aplot_3.uni_plot(1, xx + 1, np.exp(xx + 1))
        aplot_3.uni_plot(2, xx + 1, np.exp(xx + 1))
        aplot_3.uni_plot(3, xx, np.exp(xx))
        aplot_3.tight_layout()

        self.image_name = "image_uniplot_2_sharex_not_same_interval_for_x"
        self.check_plot()

    def test_uni_plot_for_plots_same_graph_sharey(self):
        # uni plot *4
        aplot_3 = APlot(how=(2, 2), sharey=True)
        aplot_3.uni_plot(0, self.xx, self.yy)
        aplot_3.uni_plot(1, self.xx, self.yy)
        aplot_3.uni_plot(2, self.xx, self.yy)
        aplot_3.uni_plot(3, self.xx, self.yy)
        aplot_3.tight_layout()

        self.image_name = "image_uniplot_2_sharey"
        self.check_plot()

    def test_uni_plot_for_plots_same_graph_sharey_not_same_interval_for_y(self):
        xx = np.linspace(-1, 1, 1000)
        aplot_3 = APlot(how=(2, 2), sharey=True)
        aplot_3.uni_plot(0, xx, np.exp(xx))
        aplot_3.uni_plot(1, xx + 1, np.exp(xx + 1))
        aplot_3.uni_plot(2, xx + 1, np.exp(xx + 1))
        aplot_3.uni_plot(3, xx, np.exp(xx))
        aplot_3.tight_layout()

        self.image_name = "image_uniplot_2_sharey_not_same_interval_for_y"
        self.check_plot()


    def test_bi_plot(self):
        aplot_4 = APlot(how=(2, 2))
        aplot_4.bi_plot(0, 1, self.xx, self.yy, self.xx, self.yy)
        aplot_4.uni_plot(2, self.xx, self.yy)
        aplot_4.uni_plot(3, self.xx, self.yy)
        aplot_4.tight_layout()

        self.image_name = "image_biplot"
        self.check_plot()

    def test_uni_plot_ax_bis(self):
        # two plots same figures
        aplot_1 = APlot()
        aplot_1.uni_plot(0, self.xx, self.yy + 5)
        aplot_1.uni_plot_ax_bis(0, self.xx, np.exp(self.xx))

        self.image_name = "image_plot_bis"
        self.check_plot()

    def test_cumulative_plot(self):
        aplot = APlot()
        values, base, _ = plt.hist(self.yy, bins=len(self.yy))
        aplot.cumulative_plot(base, values, nb_ax=0)

        self.image_name = "image_cumulative_plot"
        self.check_plot()
        pass

    def test_hist(self):
        aplot = APlot()
        aplot.hist(self.yy)
        self.image_name = "plot_hist"
        self.check_plot()

    def test_plot_function(self):
        def f(x):
            return 3 * x + 2 * x * x

        aplot = APlot()
        aplot.plot_function(f, self.xx)
        self.image_name = "image_plot_function"
        self.check_plot()


    def test_plot_vertical_line(self):
        aplot = APlot()
        aplot.plot_vertical_line(x=1, yy=np.array([2, 4]))

        self.image_name = "image_vertical_line"
        self.check_plot()


    def test_plot_line(self):
        aplot = APlot()
        aplot.plot_line(a=1, b=2, xx=self.xx)

        self.image_name = "image_line"
        self.check_plot()


    def test_plot_point(self):
        aplot = APlot()
        aplot.plot_point(x=1, y=1, dict_plot_param={'markersize': 5})

        self.image_name = "image_point"
        self.check_plot()

    def test_help_dict_plot(self):
        pass

    def test_help_dict_ax(self):
        pass
