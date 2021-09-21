import numpy as np
from priv_lib_util.tools import function_iterable

from src.nn_classes.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla
from src.nn_train.fit import nn_fit


def nn_train(net, data_X, data_Y, params_training, indic_train_X, indic_train_Y,
             early_stoppers=(Early_stopper_vanilla(),),
             indic_val_X=None, indic_val_Y=None, *, silent=False):
    """
    Semantics : Given the net, we train it upon data.
    For optimisation reasons, we pass the indices.
    Args:
        net (Savable_net):
        data_X (tensor):
        data_Y (tensor):
        params_training (NNTrainParameters): parameters used for training
        indic_train_X: indices of values from data_X to be used for training
        indic_train_Y: indices of values from data_Y to be used for training
        early_stoppers (iter Early_stopper): Used for deciding if the training should stop early.
                        Preferably immutable to insure no changes.
        indic_val_X: indices of values from data_X to be used for validation, None if validation is not performed
        indic_val_Y: indices of values from data_Y to be used for validation, None if validation is not performed
        silent (bool): verbose.

    Returns: history of training and all other metrics, in a dictionary.
             best epoch for training (int).
                    history_kfold has the form:
                    history = {'training': {},'validation': {}}
                    history['training']['loss'] = np.zeros((nb_split, parameters_training.epochs))
                    history['validation']['loss'] = np.zeros((nb_split, parameters_training.epochs))
                    for metric in parameters_training.metrics:
                        history['training'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
                        history['validation'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
            best_epoch_for_model looks like: [10,200,5]

    Post-condition :
        early_stoppers not changed.
        Net is modified in the progress.

    """

    # Prepare Training set
    device = params_training.device
    epoch = params_training.epochs
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train_on_device = data_Y[indic_train_Y].to(device)

    # condition if we use validation set:
    list_params_validation = [indic_val_X, indic_val_Y]
    is_val_included = not function_iterable.are_at_least_one_None(list_params_validation)
    #: equivalent to "are all not None ?"
    if not is_val_included:
        function_iterable.raise_if_not_all_None(list_params_validation)

    X_val_on_device, Y_val_on_device, history = _history_creation(data_X, data_Y, device, epoch, indic_val_X,
                                                                  indic_val_Y, is_val_included,
                                                                  params_training)

    if is_val_included:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, params_training, history, early_stoppers,
                                X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, silent=silent)

    else:  # if no validation set
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, params_training, history, early_stoppers,
                                silent=silent)

    return history, epoch_best_net


def _history_creation(data_X, data_Y, device, epoch, indic_validation_X, indic_validation_Y, is_validation_included,
                      params_training):
    # initialise the training history for loss and any other metric included
    history = {'training': {}}
    history['training']['loss'] = np.full(epoch, np.nan)
    for metric in params_training.metrics:
        history['training'][metric.name] = np.full(epoch, np.nan)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indic_validation_X].to(device)
        Y_val_on_device = data_Y[indic_validation_Y].to(device)

        # initialise the validation history for loss and any other metrics included
        # initialise with nans such that no plot if no value.
        history['validation'] = {}
        history['validation']['loss'] = np.full(epoch, np.nan)

        for metric in params_training.metrics:
            # initialise with nans such that no plot if no value.
            history['validation'][metric.name] = np.full(epoch, np.nan)
    else:
        X_val_on_device = 0
        Y_val_on_device = 0
    return X_val_on_device, Y_val_on_device, history
