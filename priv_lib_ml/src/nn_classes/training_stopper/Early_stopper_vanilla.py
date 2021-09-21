from src.nn_classes.training_stopper.Early_stopper import Early_stopper


class Early_stopper_vanilla(Early_stopper):
    # has the parameter has_improved_last_epoch set as true so it will save the new model.
    # it will never stopped!
    # thus do not check history parameter.
    # the call method is overridden, and thus
    def __init__(self):
        super().__init__(tipee="vanilla", metric_name="vanilla", patience=0, silent=True, delta=0.)

    def __call__(self, *args, **kwargs):
        # overload in order to always return false
        return False

    def _is_early_stop(self, history, epoch):
        return False
