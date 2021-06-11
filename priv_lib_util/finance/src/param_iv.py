# normal libraries
import numpy as np
import math

from priv_lib_util.calculus.src.optimization import newtons_method
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import warnings

# my libraries
from priv_lib_util.finance.src.BS_model import BlackScholes, BlackScholesVegaCore



phi = lambda xx: (1 - (1 - np.exp(-xx)) / xx) / xx
phi_lambda = lambda xx, lamb: phi(xx * lamb)
phi_curry = lambda lamb: lambda xx: phi_lambda(xx, lamb)

phi_power_law = lambda eta, gamma: lambda theta: eta * theta ** (- gamma)


def compute_omega_SSVI(KK, theta, rho, phi, S0, log_price=True):
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
    return theta / 2 * \
           (1 +
            rho * phi * k
            + np.sqrt(expression_in_root * expression_in_root + 1 - rho * rho)
            )


def natural_SVIparameters2density(val_density, parameters):
    """ takes natural SVI parameters."""
    w = TIV_SSVI
    w_dash = TIV_omega_SSVI_dash
    w_dash_dash = TIV_omega_SSVI_dashdash
    return TIV_to_density_litzenberg(val_density, w, w_dash, w_dash_dash, parameters)


def natural_SVIparameters2price(val_density, parameters, log_moneyness):
    """ takes natural SVI parameters."""
    density_of_SVI = natural_SVIparameters2density(val_density, parameters) * np.exp(-val_density)  # change of variable
    val_density = np.exp(val_density)  # push
    density_of_SVI_interpolated = Density(val_density, density_of_SVI)
    S0 = compute_pushed_expectation(val_density, density_of_SVI, None)
    C_k = compute_price(val_density, np.exp(log_moneyness), density_of_SVI_interpolated(val_density),
                        None)
    return density_of_SVI_interpolated, C_k, S0


def natural_SVIparameters2TIV(val_density, parameters, log_moneyness):
    """ takes natural SVI parameters."""
    density_of_SVI_interpolated, C_k, S0 = natural_SVIparameters2price(val_density, parameters, log_moneyness)

    TIV = np.zeros(len(log_moneyness))
    for j in range(len(log_moneyness)):
        TIV[j] = invert_from_call_to_TIV(log_moneyness[j], C_k[j], S0, push=np.exp)
    return density_of_SVI_interpolated, C_k, S0, TIV


def g_Jacquier(kk, w, w_dash, w_dash_dash, parametrisation):
    # k vectorised
    temp_w = w(parametrisation, kk)
    temp_w_dash = w_dash(parametrisation, kk)
    temp_w_dash_dash = w_dash_dash(parametrisation, kk)

    temp = (1 - kk * temp_w_dash / (2 * temp_w))

    return (temp * temp
            - temp_w_dash * temp_w_dash / 4 * (1 / temp_w + 0.25)
            + temp_w_dash_dash / 2)


def SVIparam_natural2raw(PARAMETRISATION):
    delta, mu, rho, omega, zeta = PARAMETRISATION
    a = delta + omega / 2 * (1 - rho * rho)
    b = omega * zeta / 2
    m = mu - rho / zeta
    sigma = math.sqrt(1 - rho * rho) / zeta
    return a, b, rho, m, sigma


def TIV_to_density_litzenberg(kk, w, w_dash, w_dash_dash, parametrisation):
    # time through parametrisation
    temp_w = w(parametrisation, kk)
    g_value = g_Jacquier(kk, w, w_dash, w_dash_dash, parametrisation)
    temp2 = np.sqrt(2 * math.pi * temp_w)
    temp_d_ = -kk / np.sqrt(temp_w) - np.sqrt(temp_w) / 2
    temp3 = np.exp(- temp_d_ * temp_d_ / 2)
    return g_value * temp3 / temp2


def TIV_SSVI(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return a + b * (rho * (kk - m) + np.sqrt(under_the_root))


def TIV_omega_SSVI_dash(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return (b * rho +
            b * (kk - m) / np.sqrt(under_the_root)
            )


def TIV_omega_SSVI_dashdash(parametrisation, kk):
    # kk vectorised
    a, b, rho, m, sigma = parametrisation
    assert b >= 0
    assert abs(rho) <= 1
    assert sigma > 0
    under_the_root = (kk - m) * (kk - m) + sigma * sigma
    return b * sigma * sigma * np.power(under_the_root, -3 / 2)
