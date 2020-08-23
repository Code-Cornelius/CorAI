import numpy as np


def newtons_method(f, df, x0, e=10 ** (-10), tol=10 ** (-10)):
    # e is the error.
    # tol is the each step tolerance

    ## f is the function
    ## df its derivative
    ## x0    first guess
    ## e the tolerance
    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 1
    while f(x0) > e or step > tol:
        if step == np.inf or number_of_step_crash > np.power(10, 9): # function too flat.
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1
        old_x0 = x0
        x0 = x0 - f(x0) / df(x0)
        step = abs(x0 - old_x0)
    return x0


def newtons_method_multi(df, ddf, x0, e=10 ** (-10), tol=10 ** (-10)):
    # e is the error.
    # tol is the each step tolerance

    ## df is the derivative
    ## ddf its Hessian
    ## x0    first guess
    ## e the tolerance
    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 1
    while np.linalg.norm(df(x0), 2) > e or step > tol: #I use norm 2 as criterea
        if number_of_step_crash > np.power(10, 9):
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1
        old_x0 = x0

        A = ddf(x0)
        A = np.linalg.inv(A)
        B = df(x0)

        x0 = x0 - np.matmul(A, B)
        step = abs(x0 - old_x0)
    return x0