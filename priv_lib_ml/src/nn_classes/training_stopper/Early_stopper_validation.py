from src.nn_classes.training_stopper.Early_stopper import Early_stopper


class Early_stopper_validation(Early_stopper):
    """     Used to stop training earlier given some criteria. Undefined behaviour when patience is greater than nb of epoch.
    """

    def __init__(self, patience=10, silent=True, delta=0.):
        """Delta is the percent of changed one needs to observe wrt lowest loss."""
        super().__init__(tipee='validation', metric_name='loss', patience=patience, silent=silent, delta=delta)

    def _is_early_stop(self, validation_losses, epoch):
        """ the critirea is whether the NN is not overfitting: i.e. the validation loss is decreasing. If delta is too big, then a model where the validation is constant keeps training !"""
        if self._lowest_loss * (1 + self._delta) > validation_losses[epoch]:
            return False
        else:
            return True
