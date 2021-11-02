import warnings

import numpy as np


def newtons_method(f, df, x0, e=1E-10, tol=1E-10):
    """
    Semantics:
        newton method for finding the root of f given its derivative.
    Args:
        f (callable):  function for finding its roots.
        df (callable):  derivative of f as a function
        x0:  initial gues.
        e (error for the root):
        tol (tol for step):

    Returns:

    """
    number_of_step_crash = 0
    step = 1
    while f(x0) > e or abs(step) > tol:
        if abs(x0) == np.inf or number_of_step_crash > 1E3:  # function too flat.
            raise ValueError("Is the function flat enough ?")
        number_of_step_crash += 1
        step = f(x0) / df(x0)
        x0 -= step
    return x0


def newtons_method_multi(f, df, x0, e=1E-10, tol=1E-10):
    """
    Semantics:
        newton method for finding the root of f given its derivative.
        f is a multi dim function R^n -> R.
    Args:
        f (callable):  function for finding its roots.
        df (callable):  derivative of f as a function, gradient.
        x0:  initial gues.
        e (error for the root):
        tol (tol for step):

    Returns:

    """
    number_of_step_crash = 0
    step = 1
    # TODO step > tol is not supposed to work. comparing vector and float. use something like any
    while np.linalg.norm(df(x0), 2) > e or abs(step) > tol:  # I use norm 2 as criterea
        if number_of_step_crash > 1E4:
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1

        A = np.linalg.inv(df(x0))
        B = f(x0)
        step = np.matmul(A, B)
        x0 -= step
    return x0


def newtons_method_vectorised(f, df, x0, e=1E-7, tol=1E-7, silent=True):
    """
    Semantics:
        vectorised newton method for finding the root of f given its derivative.
        the vectorisation means that one can give a function that is in R^d.
        All the inputs are optimised independently.
        Each needs to reach the desired precision.
        If some inputs did not converged (step reached infinity), replace the value by nans.
    Args:
        f (callable):  function for finding its roots.
        Takes two parameters, an array, and a list of indices to slice the array.
        df (callable):  derivative of f as a function.
        Takes two parameters, an array, and a list of indices to slice the array.
        x0:  initial guess. Is changed over iterations.
        e (error for the root):
        tol (tol for step):
        silent (bool): verbose.

    Returns:
        void. x0 contains the optimised values.

    """
    nb_step = 0
    arr_flags_iter = np.full(len(x0), True)

    f_eval = f(x0, arr_flags_iter)
    df_eval = df(x0, arr_flags_iter)

    arr_flags_iter = (np.abs(f_eval / df_eval) > tol) & \
                     (np.abs(f_eval) > e)

    step = np.zeros(len(x0))  # initialization
    while np.any(arr_flags_iter):  # test that they all converged, tol for step and error
        if not nb_step < 10000:
            # test if any step is infinity
            raise Exception("Is the function flat enough ?")

        # Iterate on the indices that haven't yet converged
        f_eval = f(x0, arr_flags_iter)
        step[arr_flags_iter] = f_eval / df(x0, arr_flags_iter)
        step[(step == np.inf)] = np.NaN
        x0[arr_flags_iter] = x0[arr_flags_iter] - step[arr_flags_iter]
        arr_flags_iter[arr_flags_iter] = (np.abs(step[arr_flags_iter]) > tol) & \
                                         (np.abs(f_eval) > e)

        nb_step += 1
    if not silent:
        print('converged in {it} iterations'.format(it=nb_step))
    if np.any(x0 == np.NaN):
        warnings.warn("Nans")

    return

