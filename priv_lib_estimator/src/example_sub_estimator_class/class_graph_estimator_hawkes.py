# normal libraries
# my libraries
from library_functions.tools.classical_functions_vectors import is_iterable
# errors:

# other files
from classes.class_estimator_hawkes import *
from classes.class_kernel import *


# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    SEPARATORS = ['parameter', 'm', 'n']

    def __init__(self, estimator_hawkes, fct_parameters, *args, **kwargs):
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        
        if not isinstance(estimator_hawkes, Estimator_Hawkes):
            raise Error_type_setter(f'Argument is not an {str(Estimator_Hawkes)}.')
            

        super().__init__(estimator=estimator_hawkes, fct_parameters=fct_parameters,
                         separators=Graph_Estimator_Hawkes.SEPARATORS,
                         *args, **kwargs)

        # parameters is a list of lists of lists of functions
        self.M = np.shape(fct_parameters[1])[1]
        self.ALPHA = fct_parameters[1]  # I split the ALPHA BETA AND NU instead
        self.BETA = fct_parameters[2]  # of one parameter fct_parameters because such names are more readable.
        self.NU = fct_parameters[0]
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(self.BETA)) #Here is the parameter in full line, instead of matrix 3dimensions.
        self.T_max = estimator_hawkes.DF["T_max"].max()
        self.nb_of_guesses = estimator_hawkes.DF['number of guesses'].max()

    @classmethod
    def from_path(cls, path, parameters):
        # path has to be raw. with \\
        estimator = Estimator_Hawkes()
        estimator.append(pd.read_csv(path))
        return cls(estimator_hawkes=estimator, fct_parameters=parameters)


    # section ######################################################################
    #  #############################################################################
    # getters/setters
    
    @property
    def ALPHA(self):
        return self._ALPHA

    @ALPHA.setter
    def ALPHA(self, new_ALPHA):
        if is_iterable(new_ALPHA) and all([callable(new_ALPHA[i][j])
                                           for i in range(self.M) for j in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._ALPHA = new_ALPHA
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def BETA(self):
        return self._BETA

    @BETA.setter
    def BETA(self, new_BETA):
        if is_iterable(new_BETA) and all([callable(new_BETA[i][j])
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
        if is_iterable(new_NU) and all([callable(new_NU[i]) for i in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._NU = new_NU
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def T_max(self):
        return self._T_max

    @T_max.setter
    def T_max(self, new_T_max):
        if isinstance(new_T_max, float):
                self._T_max = new_T_max
        else:
            raise Error_type_setter(f'Argument is not an {str(float)}.')


    @property
    def nb_of_guesses(self):
        return self._nb_of_guesses

    @nb_of_guesses.setter
    def nb_of_guesses(self, new_nb_of_guesses):
        # here it is tricky because th original nb_of_guesses is not an int but a numpy.int. So I have to use the test from numpy.
        if isinstance(new_nb_of_guesses, (int,np.integer)):
                self._nb_of_guesses = new_nb_of_guesses
        else:
            raise Error_type_setter(f'Argument is not an {str(int)}.')

