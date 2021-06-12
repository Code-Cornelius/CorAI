# normal libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import simps

# my libraries
from priv_lib_util.finance.src.BS_model import BlackScholes
from priv_lib_plot import APlot
from priv_lib_util.finance.src.financials import compute_price, compute_integral
from priv_lib_util.finance.src.implied_vol import implied_volatility_newton, implied_vol_bisect, \
    total_implied_vol_newton

phi_heston = lambda xx: (1 - (1 - np.exp(-xx)) / xx) / xx
phi_heston_lambda = lambda xx, lamb: phi_heston(xx * lamb)
phi_heston_curry = lambda lamb: lambda xx: phi_heston_lambda(xx, lamb)

phi_power_law = lambda eta, gamma: lambda theta: eta * theta ** (- gamma)


# section ######################################################################
#  #############################################################################
# parametrisation


def g_jacquier(kk, w, w_dash, w_dash_dash, parametrisation):
    # k vectorised
    temp_w = w(parametrisation, kk)
    temp_w_dash = w_dash(parametrisation, kk)
    temp_w_dash_dash = w_dash_dash(parametrisation, kk)

    temp = (1 - kk * temp_w_dash / (2 * temp_w))

    return (temp * temp
            - temp_w_dash * temp_w_dash / 4 * (1 / temp_w + 0.25)
            + temp_w_dash_dash / 2)


def SVIparam_natural2raw(parametrisation):
    delta, mu, rho, omega, zeta = parametrisation
    a = delta + omega / 2 * (1 - rho * rho)
    b = omega * zeta / 2
    m = mu - rho / zeta
    sigma = math.sqrt(1 - rho * rho) / zeta
    return a, b, rho, m, sigma


def total_implied_vol2density_litzenberg(kk, w, w_dash, w_dash_dash, parametrisation):
    # time through parametrisation
    temp_w = w(parametrisation, kk)
    g_value = g_jacquier(kk, w, w_dash, w_dash_dash, parametrisation)
    temp2 = np.sqrt(2 * math.pi * temp_w)
    temp_d_ = -kk / np.sqrt(temp_w) - np.sqrt(temp_w) / 2
    temp3 = np.exp(- temp_d_ * temp_d_ / 2)
    return g_value * temp3 / temp2


def total_implied_vol_ssvi(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return a + b * (rho * (kk - m) + np.sqrt(under_the_root))


def total_implied_vol_ssvi_dash(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return (b * rho + b * (kk - m) / np.sqrt(under_the_root))


def total_implied_vol_ssvi_dash_dash(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return b * sigma * sigma * np.power(under_the_root, -3 / 2)


# section ######################################################################
#  #############################################################################
#  SSVI


def compute_total_implied_vol_SSVI(KK, theta, rho, phi, S0, log_price=True):
    # computes for all [rhos,theta,phi(theta)] * K the SSVI

    # K length nb of strikes
    # UU theta and rho same length
    # all numpy arrays

    if log_price:
        k = KK
    else:
        k = np.log(KK / S0)
    phi = phi(theta)

    k = np.repeat(k[None, :], len(theta), axis=0)
    theta = np.repeat(theta[:, None], len(KK), axis=1)
    rho = np.repeat(rho[:, None], len(KK), axis=1)
    phi = np.repeat(phi[:, None], len(KK), axis=1)

    expression_in_root = (phi * k + rho)
    return theta / 2 * (1 + rho * phi * k
                        + np.sqrt(expression_in_root * expression_in_root + 1 - rho * rho)
                        )


def natural_SVIparam2density(xx_for_density, parameters):
    # """ takes natural SVI parameters.  """
    """
    Semantics:
        From

    Args:
        xx_for_density:
        parameters:

    Returns:

    """
    w = total_implied_vol_ssvi
    w_dash = total_implied_vol_ssvi_dash
    w_dash_dash = total_implied_vol_ssvi_dash_dash
    return total_implied_vol2density_litzenberg(xx_for_density, w, w_dash, w_dash_dash, parameters)


def natural_SVIparameters2price(log_asset_for_density, parameters, log_moneyness):
    """ takes natural SVI parameters."""
    values_density_of_SVI = natural_SVIparam2density(log_asset_for_density, parameters) * np.exp(-val_density)
    asset_for_density = np.exp(log_asset_for_density)  # density of S_T
    s0 = compute_integral(asset_for_density, values_density_of_SVI)
    c_k = compute_price(asset_for_density, np.exp(log_moneyness), values_density_of_SVI)
    return values_density_of_SVI, c_k, s0


def natural_SVIparameters2TIV(val_density, parameters, log_moneyness):
    """ takes natural SVI parameters."""
    values_density_of_SVI, c_k, s0 = natural_SVIparameters2price(val_density, parameters, log_moneyness)
    sigma = implied_volatility_newton(True, s0, np.exp(log_moneyness), 1, 0, 0, c_k)
    total_implied_vol = 1 * sigma * sigma
    total_implied_vol = total_implied_vol_newton(True, s0, np.exp(log_moneyness), 0, 0, c_k)

    return values_density_of_SVI, c_k, s0, total_implied_vol


# section ######################################################################
#  #############################################################################
# TEST

if __name__ == '__main__':

    param1 = (0.2, -0.5)
    param2 = (0.3, -0.5)

    param_dens_1 = 0, 0, -0.5, 0.2, phi_heston_curry(1)(0.2)
    param_dens_2 = 0, 0, -0.5, 0.3, phi_heston_curry(1)(0.3)

    NB_POINTS_INTERP_DENS = int(1 * 1E5)
    NB_OF_UU = 2
    NB_POINTS_CDF_PDF = 10000
    NB_LOG_MONEYNESS = 200
    max_log_money = 0.3
    numerical_support_for_density = -6, math.log(12)

    # section ######################################################################
    #  #############################################################################
    # DENSITY
    val_density = np.linspace(*numerical_support_for_density, NB_POINTS_INTERP_DENS)
    log_moneyness = np.linspace(-max_log_money, max_log_money, NB_LOG_MONEYNESS)

    special_val_density = np.linspace(-0.3, 0.3, NB_POINTS_INTERP_DENS)
    special_log_moneyness = np.linspace(-1.5, 1.5, NB_LOG_MONEYNESS)
    temp_w = total_implied_vol_ssvi((-0.0410, 0.1331, 0.3060, 0.3586, 0.4153), special_log_moneyness)
    g_value = g_jacquier(special_log_moneyness,
                         total_implied_vol_ssvi,
                         total_implied_vol_ssvi_dash,
                         total_implied_vol_ssvi_dash_dash,
                         (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153))
    APlot().uni_plot(nb_ax = 0, xx= special_log_moneyness, yy =temp_w, dict_ax = {'title':'$\omega$'})
    APlot().uni_plot(nb_ax = 0, xx=special_log_moneyness, yy=g_value, dict_ax={'title': 'g'})

    special_val_density = np.linspace(-3, 3, NB_POINTS_INTERP_DENS)
    special_log_moneyness = np.linspace(-0.5, 2.5, NB_LOG_MONEYNESS)
    density_of_SVI, C_k, S0, TIV = natural_SVIparameters2TIV(special_val_density,
                                                                          (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153),
                                                                          special_log_moneyness)
    APlot(datax=special_val_density, datay=density_of_SVI)
    APlot(datax=special_log_moneyness, datay=TIV)
    True_TIV = total_implied_vol_ssvi((-0.0410, 0.1331, 0.3060, 0.3586, 0.4153), special_log_moneyness)
    APlot(datax=special_log_moneyness, datay=True_TIV)

    # solutions
    true_TIV1 = total_implied_vol_ssvi(SVIparam_natural2raw(param_dens_1), log_moneyness)
    true_TIV2 = total_implied_vol_ssvi(SVIparam_natural2raw(param_dens_2), log_moneyness)

    values_density_of_SVI1, C_k1, S01, TIV1 = natural_SVIparameters2TIV(val_density,
                                                                        SVIparam_natural2raw(param_dens_1),
                                                                        log_moneyness)
    values_density_of_SVI2, C_k2, S02, TIV2 = natural_SVIparameters2TIV(val_density,
                                                                        SVIparam_natural2raw(param_dens_2),
                                                                        log_moneyness)
    val_density = np.exp(val_density)
    integral_of_pdf_1 = simps(values_density_of_SVI1, val_density)
    integral_of_pdf_2 = simps(values_density_of_SVI2, val_density)

    price_1 = compute_integral(val_density, values_density_of_SVI1)
    price_2 = compute_integral(val_density, values_density_of_SVI2)
    print("ST1 = ", price_1)
    print("ST2 = ", price_2)
    print("Relative Error between both prices: ", (price_2 - price_1) / min(price_2, price_1) * 100, "%")

    uu = np.linspace(1, 2, NB_OF_UU)

    densities_plot = APlot(how=(1, 1))
    densities_plot.uni_plot(nb_ax=0, xx=val_density[::100], yy=values_density_of_SVI1[::100],
                            dict_plot_param={"color": "red", "label": "Density 1"})
    densities_plot.uni_plot(nb_ax=0, xx=val_density[::100], yy=values_density_of_SVI2[::100],
                            dict_plot_param={"label": "Density 2"},
                            dict_ax={"title": "Densities", "xlabel": "Log Moneyness", "ylabel": "PDF"})
    densities_plot.show_legend()

    true_vs_computed_TIV = APlot(how=(1, 1))
    true_vs_computed_TIV.uni_plot(nb_ax=0, xx=log_moneyness, yy=true_TIV1,
                                  dict_plot_param={"label": "1 direct", "color": "green", "linestyle": "--",
                                                   "linewidth": 3})
    true_vs_computed_TIV.uni_plot(nb_ax=0, xx=log_moneyness, yy=true_TIV2,
                                  dict_plot_param={"label": "2 direct", "color": "orange", "linestyle": "--",
                                                   "linewidth": 3})
    true_vs_computed_TIV.uni_plot(nb_ax=0, xx=log_moneyness, yy=TIV1,
                                  dict_plot_param={"label": "1 through", "color": "blue", "linewidth": 1})
    true_vs_computed_TIV.uni_plot(nb_ax=0, xx=log_moneyness, yy=TIV2,
                                  dict_plot_param={"label": "2 through", "color": "red", "linewidth": 1})

    true_vs_computed_TIV.show_legend()
    APlot.show_and_continue(2)

    sigma_to_input = np.append(true_TIV1, true_TIV2).reshape(2, 200)


    APlot.show_plot()
