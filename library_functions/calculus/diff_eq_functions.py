import numpy as np
import math

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