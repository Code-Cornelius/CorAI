from priv_lib_estimator import Estimator


class Estim_time(Estimator):
    NAMES_COLUMNS = {'TIME'}

    def __init__(self):
        super().__init__()