import time

import corai
import numpy as np
import torch
from corai import Estim_history
# inspired from corai https://github.com/Code-Cornelius/corai
from tqdm import tqdm


def history_create(nb_epochs_total, metrics):
    # initialise the training history for loss and any other metric included
    history = {'training': {}}
    for metric in metrics:
        history['training'][metric.name] = np.full(nb_epochs_total, np.nan)
        # initialise the validation history for loss and any other metrics included
        # initialise with nans such that no plot if no value.
        history['validation'] = {}
        for metric in metrics:
            # initialise with nans such that no plot if no value.
            history['validation'][metric.name] = np.full(nb_epochs_total, np.nan)

    return history


def update_history(net, metrics, epoch, is_valid_included, total_number_data, train_loader_on_device,
                   validat_loader_on_device, history):
    # update the history by adding the computed metrics.
    # one cannot compute the prediction only once. Because of encapsulation,
    # it is not obvious whether the data needs to be on device or cpu.
    ######################
    # Training Metrics   #
    ######################
    for metric in metrics:
        _update_history_for_metric(metric, net, epoch, total_number_data, history, train_loader_on_device, 'training')

    ######################
    #   Validation Loss  #
    ######################
    # the advantage of computing it in this way is that we can load data while
    if is_valid_included:
        for metric in metrics:
            _update_history_for_metric(metric, net, epoch, total_number_data, history,
                                       validat_loader_on_device, 'validation')

    return


def _update_history_for_metric(metric, net, epoch, total_number_data, history, data_loader, type):
    history[type][metric.name][epoch] = 0
    # for batch_X, batch_y in data_loader:
    #     history[type][metric.name][epoch] += metric(net, batch_X, batch_y)
    history[type][metric.name][epoch] += metric(net, data_loader[0], data_loader[1])

    history[type][metric.name][epoch] /= total_number_data[0] if type == 'training' else total_number_data[1]


def translate_history_to_dataframe(history, fold_number, validation, nb_epochs):
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
    translated_history['epoch'] = [*range(nb_epochs)]

    # add the fold number to the history
    translated_history['fold'] = [fold_number] * nb_epochs

    return translated_history


def generate_estims_history(hyper_params, input_train, output_train, input_test, output_test,
                            config_architecture, linker_estims, linker_models,
                            metrics, silent):
    estims = []
    nb_epochs = 1
    nb_prediction = 1, 1  # train , val
    for i, params in enumerate(tqdm(hyper_params)):
        current_model = config_architecture(params, input_train.shape[2], output_train.shape[2])

        corai.set_seeds(params["seed"])  # set seed for pytorch.
        ########################################################### training
        start_train_fold_time = time.time()
        _ = current_model(input_train, output_train)
        end_train_fold_time = time.time() - start_train_fold_time

        ########################################################### error comput
        predicted_test = current_model.nn_predict(input_test)[:, 0]
        true_prediction_test = \
            output_test[current_model.washout:, 0, 0].unsqueeze(1)  # missing dimension from slicing
        mse = torch.nn.functional.mse_loss(predicted_test, true_prediction_test)
        if not silent:
            print("MSE error on testing data: ", mse.item())

        ########################################################### saving the results
        estimator_history = Estim_history(metric_names=[metric.name for metric in metrics],
                                          validation=True, hyper_params=params)
        # initialise the training history for loss and any other metric included
        history = history_create(nb_epochs, metrics)
        update_history(current_model, metrics, nb_epochs - 1, True, nb_prediction,
                       (input_train, output_train), (input_test, output_test), history)
        df_history = translate_history_to_dataframe(history=history, fold_number=0,
                                                    validation=estimator_history.validation, nb_epochs=nb_epochs)
        estimator_history.append(history=df_history, fold_best_epoch=0, fold_time=end_train_fold_time)
        estimator_history.best_fold = 0

        estims.append(estimator_history)  # add to the list of estimators
        ########################################################### saving
        estimator_history.to_json(path=linker_estims([f"estim_{i}.json"]), compress=False)
        current_model.save_net(path=linker_models([f"model_{i}.pth"]))
    return estims
