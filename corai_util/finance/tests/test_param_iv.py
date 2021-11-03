from scipy.integrate import simps

from corai_plot import APlot
from corai_util.finance.src.param_iv import *

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
    APlot().uni_plot(nb_ax=0, xx=special_log_moneyness, yy=temp_w, dict_ax={'title': '$\omega$'})
    APlot().uni_plot(nb_ax=0, xx=special_log_moneyness, yy=g_value, dict_ax={'title': 'g'})

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
