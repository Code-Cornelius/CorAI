from priv_lib_error import Error_type_setter
from priv_lib_estimator import Plot_estimator

from nn_classes.estimator.hyper_parameters.estim_hyper_param import Estim_hyper_param


class Plot_estim_hyper_param(Plot_estimator):

    def __init__(self, estim_hyper_param, *args, **kwargs):
        if not isinstance(estim_hyper_param, Estim_hyper_param):
            raise Error_type_setter(f'Argument is not an {str(Estim_hyper_param)}.')
        super().__init__(estim_hyper_param, *args, **kwargs)