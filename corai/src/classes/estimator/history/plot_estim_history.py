from corai_estimator import Plot_estimator


class Plot_estim_history(Plot_estimator):

    def __init__(self, estimator_hist, *args, **kwargs):
        # typechecking is cumbersome in big projects.

        # if not isinstance(estimator_bench, Estim_history):
        #     raise Error_type_setter(f'Argument is not an {str(Estim_history)}.')
        super().__init__(estimator=estimator_hist, *args, **kwargs)
