# normal libraries

# priv_libraries
import numpy as np
import pandas as pd
from priv_lib_estimator import Estimator


# other files


# section ######################################################################
#  #############################################################################
# class


class Estim_hawkes(Estimator):
    CORE_COL = {'parameter', 'n', 'm', 'time estimation',
                'weight function', 'value', 'T_max', 'time_burn_in',
                'true value', 'number of guesses'}
    # m is the column
    # n is the line

    def __init__(self, df=None, *args, **kwargs):
        super().__init__(df=df, *args, **kwargs)

    def mean(self, separator=None):
        """

        Args:
            separator: is a list, of the estimators to gather together.

        Returns: the output format is list of lists with on each line [ans_N, ans_A, ans_B],
        and on every single additional dimension, the separator.

        """

        separators = ['parameter', 'm', 'n']
        M = self.DF["m"].max() + 1
        ans_dict = {}

        # if separator is not None:
        #     for str in separator: separators.append(str)

        global_dict, keys = self.groupby_DF(separator)
        for key in keys:
            data = global_dict.get_group(key)
            dict_of_means = data.groupby(separators)['value'].mean()
            ans_N, ans_A, ans_B = [], [], []

            for i in range(M):
                ans_N.append(dict_of_means[('nu', i, 0)])
                for j in range(M):
                    if not j:  # if j == 0
                        ans_A.append([])
                        ans_B.append([])
                    # we append to this new small list the j's.
                    ans_A[i].append(dict_of_means[('alpha', i, j)])
                    ans_B[i].append(dict_of_means[('beta', i, j)])
            # i get triple list like usually.
            ans_dict[key] = [ans_N, ans_A, ans_B]
        return ans_dict


    def append(self, alpha, beta, nu, dim, T_max, kernel, times_of_estimation, time_burn_in, nb_simul):
        """
        Semantics:
            Append information from the estimation to the estimator
        Args:

        Returns:
            Void
        """
        alpha_flat = alpha.flatten() # if alpha is in format  (nb_time_estim, matrix dim * dim)
        beta_flat = beta.flatten()
        nu_flat = nu.flatten()


        nb_time_estim = len(times_of_estimation)
        times_rounded = np.around(times_of_estimation, 8)
        # wip time estim is wrong: time estim is repeated but should round robin, right now 0 0 100 100 200 200,
        # but we want 0 100 200, 0 100 200
        time_estimation = np.tile(np.repeat(times_rounded, dim + dim * dim * 2, axis = 0), nb_simul)
        # the time_estimation is rounded because sometimes the registered number is not exactly correct.

        estimation = np.concatenate([alpha_flat, beta_flat, nu_flat], axis=0)

        true_value_nu = [0] * dim * nb_time_estim * nb_simul
        true_value_alpha = [0] * dim * dim * nb_time_estim * nb_simul
        true_value_beta = [0] * dim * dim * nb_time_estim * nb_simul

        # WIP WHAT IS NU ALPHA BETA (spoiler alert: true parameters)
        for k in range(nb_time_estim):
            for l in range(nb_simul):
                for i in range(dim):
                    true_value_nu[k * nb_simul * dim + l * dim + i] = 0 #self.nu[i](times_rounded[k])
                    for j in range(dim):
                        true_value_alpha[k * nb_simul * dim * dim + l * dim * dim + i * dim + j] = 1 # self.alpha[i][j](times_rounded[k])
                        true_value_beta[k * nb_simul * dim * dim + l * dim * dim + i * dim + j] = 2 # self.beta[i][j](times_rounded[k])
        true_value = np.concatenate([np.array(true_value_alpha), np.array(true_value_beta), np.array(true_value_nu)], axis = 0)

        # wip same problem as time estimation, too little betas, too many nus, alpha is good
        # though order beta nus is right
        parameters_name = np.concatenate([np.repeat("alpha", len(alpha_flat)),
                                          np.repeat("beta", len(beta_flat)),
                                          np.repeat("nu", len(nu_flat))], axis=0)
        parameters_name = np.tile(parameters_name, nb_simul)

        # wip some problem as above, use tile, repeat alpha beta by number of estim
        nn_alpha_index_pattern = np.tile(np.arange(dim).reshape((1, dim)).transpose(), (1, dim)).flatten()
        nn_alpha_index_pattern = np.tile(nn_alpha_index_pattern, nb_time_estim)
        nn_nu_index_pattern = np.tile(np.arange(dim), nb_time_estim)
        nn = np.concatenate([nn_alpha_index_pattern, nn_alpha_index_pattern, nn_nu_index_pattern], axis=0)
        nn = np.tile(nn, nb_simul)

        mm_alpha_index_pattern = np.tile(np.arange(dim), (dim, 1)).flatten()
        mm_alpha_index_pattern = np.tile(mm_alpha_index_pattern, nb_time_estim)
        mm_nu_index_pattern = np.repeat(0, dim * nb_time_estim)
        mm = np.concatenate([mm_alpha_index_pattern, mm_alpha_index_pattern, mm_nu_index_pattern], axis=0)
        mm = np.tile(mm, nb_simul)


        # todo parameter out of the constant?
        estimator_dict_form = {
            "time estimation": time_estimation,
            "parameter": parameters_name,
            "n": nn,
            "m": mm,
            "weight function": [kernel.name] * (dim + dim * dim * 2) * nb_time_estim * nb_simul,
            "value": estimation,
            'T_max': [T_max] * (dim + dim * dim * 2) * nb_time_estim * nb_simul,
            'time_burn_in': [time_burn_in] * (dim + dim * dim * 2) * nb_time_estim * nb_simul,
            'true value': true_value,
            # the time_estimation is rounded because sometimes the registered number is not exactly correct.
            'number of guesses': [nb_time_estim] * (dim + dim * dim * 2) * nb_time_estim * nb_simul
        }
        estimator_df_form = pd.DataFrame(estimator_dict_form)
        super().append(estimator_df_form)
        return