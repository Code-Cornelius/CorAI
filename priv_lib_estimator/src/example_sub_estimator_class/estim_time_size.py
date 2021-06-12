from priv_lib_estimator.src.example_sub_estimator_class.estim_time_base import Estim_time


class Estim_time_size(Estim_time):
    NAMES_COLUMNS = Estim_time.NAMES_COLUMNS.copy()
    NAMES_COLUMNS.add('N')

    def __init__(self):
        super().__init__()
