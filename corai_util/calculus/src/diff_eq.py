import numpy as np
import math

from corai_util.tools import function_iterable

# defines the coefficients for fractional ADAMS method in order to compute a SDE path.
# it needs the number of coefficients as well as the alpha of roughness.
def fractional_ADAMS(k, alpha, DELTA):
    # a needs k+2 elements, because j \in 0,k+1.
    # b doesn't start at 0, so only k+1 elements.
    a = np.zeros(k + 2)
    b = np.zeros(k + 1)
    for i in range(k + 2):
        if i == 0:
            a[i] = k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha
        if i == k + 1:
            a[i] = 1
        if (i != k + 1) and (i != 0):
            a[i] = (k - i + 2) ** (alpha + 1) + (k - i) ** (alpha + 1) - 2 * (k - i + 1) ** (alpha + 1)
        if i != k + 1:
            b[i] = (k + 1 - i) ** alpha - (k - i) ** alpha
    a = 1 / math.gamma(alpha + 2) * DELTA ** alpha * a
    b = 1 / math.gamma(alpha + 1) * DELTA ** alpha * b
    return a, b


def system_ODE_solver(tt, starting_point, function_evolution, left_or_right="left"):
    """
    Semantics:
        ODE solver with Euler's method. It solves liner systems of ODEs of the first oder.
        left to right flow.

    Args:
        tt: grid of times. Assume regularity for the grid. The same grid for all functions.
        starting_point: LIST of lower boundary, starting condition
                        time and previous points for each function.
                        Same order as in function_evolution.
        function_evolution: LIST of "1 + len(function_evolution)" dimensional function.
        left_or_right: starting point on left or right. Changes the equations, either backwards of forwards equations.


    Notes:
        function_evolution and starting_point as lists type for optimization of the code
        (access constant and small overhead)

    Returns:
        A list with the values of the function, over the grid tt. format: [[x1,y1...], [x2,y2...] ... ]

    """
    assert function_iterable.is_iterable(function_evolution), "Function Evolution is not iterable."
    assert function_iterable.is_iterable(starting_point), "Starting Point is not iterable."

    L = len(tt)
    J = len(function_evolution)
    yy = [[0] * J for _ in range(L)]  # creates a L by J matrix.
    # List comprehension in order to avoid copy of mutable element

    if left_or_right == "left":
        yy[0] = starting_point
        a = 0  # beg range
        b = L - 1  # end range; skipping first step
        step = 1  # how increment
        DELTA = tt[1] - tt[0]

    else:
        yy[-1] = starting_point
        a = L - 1  # beg range
        b = 0  # end range; skipping first step
        step = -1  # how increment
        DELTA = tt[0] - tt[1]  # delta changes because of the direction

    for i in range(a, b, step):
        for j in range(J):
            yy[i + step][j] = yy[i][j] + DELTA * function_evolution[j](tt[i], *yy[i])
    return yy


"""

def support(sigma1, sigma2):  # gives back when the normal given by sigma 1 is lower than the normal given by sigma2
    xm = np.sqrt(2) * sigma1 * sigma2 * np.sqrt(np.log(sigma1 / sigma2) / (sigma1 - sigma2)) / np.sqrt(sigma1 + sigma2)
    return -xm, xm  # by symmetry


################ DATA:
SIGMA = [1., 2.2]
tm, tM = support(SIGMA[0], SIGMA[1])
tm *= 0.95
tM *= 0.95
# times 0.9 so one can be inside the interval, all the time.
# Otherwise we face boundary issues.
NB_GRID_POINT = int(1E4)  # conversion float to int for linspace.
tt = np.linspace(tm, tM, NB_GRID_POINT)

density_mu = lambda tt: norm.pdf(tt, 0, SIGMA[0])  # probably not the fastest choice for single evaluation, but great for vectorization !
density_nu = lambda tt: norm.pdf(tt, 0, SIGMA[1])  # probably not the fastest choice for single evaluation, but great for vectorization !

# for x \ihow_much_rotate E !
density_eta = lambda tt: np.maximum(density_mu(tt) - density_nu(tt), 0)
density_gamma = lambda tt: np.maximum(density_nu(tt) - density_mu(tt), 0)

# FIRST PLOT:
plt.figure(figsize=(10, 4))
plt.plot(tt, density_mu(tt), 'b', label=r'$\mu$')
plt.plot(tt, density_nu(tt), 'k', label=r'$\nu$')
plt.plot(tt, density_eta(tt), 'r', label=r'$\eta$')
plt.plot(tt, density_gamma(tt), 'g', label=r'$\gamma$')
plt.title("The support of E is (%.3f,%.3f)" % (support(SIGMA[0], SIGMA[1])))
plt.legend(loc="best")

# SOLVING ODE FROM LEFT !!!
func_p = lambda t, x, y: (t - y) / (y - x) * density_eta(t) / density_gamma(x)
func_q = lambda t, x, y: (x - t) / (y - x) * density_eta(t) / density_gamma(y)
STARTING_POINT = [tm / 0.95 * 1.05,  # tm /0.95 is equal to the true tm (we narrowed the interval for simulation)
                  # and *1.05 so we are not at the boundary.
                  8.]  # p(a), q(a), remember p < q. For that reason we take q(a) large (\approx to inf).

my_nested_results = system_ODE_solver(tt, STARTING_POINT, [func_p, func_q], "left")

"""
