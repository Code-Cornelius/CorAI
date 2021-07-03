# normal libraries
import numpy as np
import pandas as pd
# errors:
from priv_lib_error import Error_type_setter
# priv_libraries
from priv_lib_estimator import Plot_estimator
from priv_lib_estimator.src.example_hawkes_estim.estim_hawkes import Estim_hawkes
from priv_lib_util.tools import function_iterable


# other files


# batch_estimation is one dataframe with the estimators.
class Plot_estim_hawkes(Plot_estimator):
    CORE_COL = ['parameter', 'm', 'n']

    def __init__(self, estimator_hawkes, fct_parameters,
                 number_of_estimations, range_estimation,
                 *args, **kwargs):
        # TODO 23/06/2021 nie_k: IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.

        if not isinstance(estimator_hawkes, Estim_hawkes):
            raise Error_type_setter(f'Argument is not an {str(Estim_hawkes)}.')

        super().__init__(estimator=estimator_hawkes, fct_parameters=fct_parameters,
                         grouping_by=Estim_hawkes.CORE_COL,
                         *args, **kwargs)

        # fct_parameters = (fct_mu, fct_alpha, fct_beta) where
        # fct_nu = vector(N),
        # fct_alpha = matrix(N*N),
        # fct_beta = matrix(N * N).
        # all callable

        # parameters is a list of lists of lists of functions
        self.M = len(fct_parameters[0])
        self.NU = fct_parameters[0]
        self.ALPHA = fct_parameters[1]  # I split the ALPHA BETA AND NU instead
        self.BETA = fct_parameters[2]  # of one parameter fct_parameters because such names are more readable.
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(
            self.BETA))  # Here is the parameter in full line, instead of matrix 3dimensions.

        # wip wth is this
        self.range_estimation = range_estimation
        self.number_of_estimations = number_of_estimations

    @classmethod
    def from_path(cls, path, parameters):
        # path has to be raw. with \\
        estimator = Estim_hawkes(pd.read_csv(path))
        return cls(estimator_hawkes=estimator, fct_parameters=parameters) # WIP THIS IS WRONG


    # section ######################################################################
    #  #############################################################################
    # getters/setters

    @property
    def ALPHA(self):
        return self._ALPHA

    @ALPHA.setter
    def ALPHA(self, new_ALPHA):
        if function_iterable.is_iterable(new_ALPHA) and all([callable(new_ALPHA[i][j])
                                                             for i in range(self.M) for j in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._ALPHA = new_ALPHA
        else:
            raise Error_type_setter(f'alpha is not an function.')

    @property
    def BETA(self):
        return self._BETA

    @BETA.setter
    def BETA(self, new_BETA):
        if function_iterable.is_iterable(new_BETA) and all([callable(new_BETA[i][j])
                                                            for i in range(self.M) for j in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._BETA = new_BETA
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def NU(self):
        return self._NU

    @NU.setter
    def NU(self, new_NU):
        if function_iterable.is_iterable(new_NU) and all([callable(new_NU[i]) for i in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._NU = new_NU
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def nb_of_guesses(self):
        return self._nb_of_guesses

    @nb_of_guesses.setter
    def nb_of_guesses(self, new_nb_of_guesses):
        # here it is tricky because th original nb_of_guesses is not an int but a numpy.int. So I have to use the test from numpy.
        if isinstance(new_nb_of_guesses, (int, np.integer)):
            self._nb_of_guesses = new_nb_of_guesses
        else:
            raise Error_type_setter(f'Argument is not an {str(int)}.')

    @property
    def range_estimation(self):
        return self._range_estimation

    @range_estimation.setter
    def range_estimation(self, new_range_estimation):
        # checks in order:
        # being a tuple, length == 2, elements are ints or floats.
        if isinstance(new_range_estimation, tuple):
            if len(new_range_estimation) != 2:
                raise Error_type_setter(f"range_estimation must be length 2.")
            for elmt in new_range_estimation:
                if not isinstance(elmt, (int, float)):
                    raise Error_type_setter(f"range_estimation's elements must be floats or integers.")
                self._range_estimation = new_range_estimation
        else:
            raise Error_type_setter(f"range_estimation is not an {str(tuple)}.")
