# normal libraries
import math

import numpy as np
# priv_libraries
from priv_lib_util.finance.src.financials import compute_price, compute_integral
from priv_lib_util.finance.src.implied_vol import implied_volatility_newton, total_implied_vol_newton

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
    values_density_of_SVI = natural_SVIparam2density(log_asset_for_density, parameters) * np.exp(-log_asset_for_density)
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
