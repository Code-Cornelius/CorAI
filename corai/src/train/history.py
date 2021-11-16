import numpy as np

from corai.src.classes.estimator.history.estim_history import Estim_history


def history_create(nb_epochs_total, metrics, is_validation_included):
    # initialise the training history for loss and any other metric included
    history = {'training': {}}
    history['training']['loss'] = np.full(nb_epochs_total, np.nan)
    for metric in metrics:
        history['training'][metric.name] = np.full(nb_epochs_total, np.nan)

        if is_validation_included:
            # initialise the validation history for loss and any other metrics included
            # initialise with nans such that no plot if no value.
            history['validation'] = {}
            history['validation']['loss'] = np.full(nb_epochs_total, np.nan)

            for metric in metrics:
                # initialise with nans such that no plot if no value.
                history['validation'][metric.name] = np.full(nb_epochs_total, np.nan)
    return history


def translate_history_to_dataframe(history, fold_number, validation):
    """
        Translate from history structure to a flat structure that will be used to add the history to the dataframe
    Args:
        history: the history of the training.
        fold_number (int): the fold number the history corresponds to.
        validation (bool): flag to specify weather validation is used.
        validation (bool): is there a validation metric computed.

    Returns:
        The translated history
    """
    # TODO 26/07/2021 nie_k:  a good idea would be to slice the data wrt criterea, like only 1/25 of the epochs saved.
    translated_history = {}

    # collect training information
    for key, value in history['training'].items():
        new_key = Estim_history.generate_column_name(key)
        new_value = value[~np.isnan(value)]  # we slice out the value that are nan.
        translated_history[new_key] = new_value.tolist()

    assert ('validation' in history) == validation, "Validation required / not required and " \
                                                    "validation not present / present in the history."
    # collect validation information if present
    if 'validation' in history:
        for key, value in history['validation'].items():
            new_key = Estim_history.generate_column_name(key, validation=True)
            new_value = value[~np.isnan(value)]
            translated_history[new_key] = new_value.tolist()

    # add the epoch number to the translated history
    nb_epochs = len(translated_history['loss_training'])
    translated_history['epoch'] = [*range(nb_epochs)]

    # add the fold number to the history
    translated_history['fold'] = [fold_number] * nb_epochs

    return translated_history
