from unittest import TestCase

import numpy as np
from scipy.stats import norm, uniform, lognorm

from priv_lib_util.calculus import diff_eq
from priv_lib_util.tools import function_iterable


class Test_diff_eq(TestCase):
    def test_fractional_adams(self):
        pass

    def test_system_ode_solver(self):
        # example taken from the paper Hobson Klimmek 2015

        UNIFORM_SUPP = [[-1., 1.],
                        [-2.,
                         2.]]  # UNIFORM_SUPP[0][0] <= UNIFORM_SUPP[1][0] <= UNIFORM_SUPP[1][1] <= UNIFORM_SUPP[1][0]

        density_1 = lambda tt: uniform.pdf(tt,
                                           loc=UNIFORM_SUPP[0][0],
                                           scale=UNIFORM_SUPP[0][1] - UNIFORM_SUPP[0][0])
        density_2 = lambda tt: uniform.pdf(tt,
                                           loc=UNIFORM_SUPP[1][0],
                                           scale=UNIFORM_SUPP[1][1] - UNIFORM_SUPP[1][0])

        def density_mu(tt):
            return density_1(tt)

        def density_nu(tt):
            return density_2(tt)

        def density_eta(tt):
            return np.maximum(density_mu(tt) - density_nu(tt), 0)

        def density_gamma(tt):
            return np.maximum(density_nu(tt) - density_mu(tt), 0)

        def p_dash_open_formula(tt, xx, yy):
            return (tt - yy) / (yy - xx) * density_eta(tt) / density_gamma(xx)

        def q_dash_open_formula(tt, xx, yy):
            return (xx - tt) / (yy - xx) * density_eta(tt) / density_gamma(yy)

        tt = np.linspace(-1 * 0.999, 0.5, 1000)
        starting_points = [[1.99, -1.01], [1.01, -1.99]]
        # forward equation
        empirical = diff_eq.system_ODE_solver(tt, starting_points[0],
                                              [p_dash_open_formula, q_dash_open_formula],
                                              left_or_right="left")
        q, p = zip(*empirical)
        p = function_iterable.replace_nans_numpy(np.array(p))
        q = function_iterable.replace_nans_numpy(np.array(q))

        true_p = lambda tt: -1 / 2 * (np.sqrt(12. - 3. * tt * tt) + tt)
        true_q = lambda tt: 1 / 2 * (np.sqrt(12. - 3. * tt * tt) - tt)
        error = np.mean(np.abs(function_iterable.replace_nans_numpy(p) - true_p(tt)))
        error += np.mean(np.abs(function_iterable.replace_nans_numpy(q) - true_q(tt)))

        # backward equation
        tt = np.linspace(-0.5, 1 * 0.999, 2000)
        # forward equation
        empirical = diff_eq.system_ODE_solver(tt, starting_points[1],
                                              [p_dash_open_formula, q_dash_open_formula],
                                              left_or_right="left")
        q, p = zip(*empirical)
        p = function_iterable.replace_nans_numpy(np.array(p))
        q = function_iterable.replace_nans_numpy(np.array(q))
        error += np.mean(function_iterable.replace_nans_numpy(p) - true_p(tt))
        error += np.mean(function_iterable.replace_nans_numpy(q) - true_q(tt))

        assert error < 0.1
