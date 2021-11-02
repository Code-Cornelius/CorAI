if __name__ == '__main__':

    def bs_formula_C(r, sigma, s, t, K):
        # Value of an European call option
        if t == 0:
            t = 1e-10  # Replace by very small value to avoid runtime warning
        d1 = (np.log(s / K) + (0.5 * sigma ** 2 + r) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        P = s * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        return P


    N = 80
    Call = True

    # either s0 or K vectorised.
    K = np.linspace(0.3, 1.6, N)
    s0 = 1

    T = 3.
    R = 0.001
    d = 0.  # d does not work since I do not have implemented vegacore for d > 0... but well the intention is there :)

    sigma = 0.1
    print(
        implied_volatility_newton(True, s0, K, T, R, d,
                                  bs_formula_C(R, sigma, s0, T, K))
    )
