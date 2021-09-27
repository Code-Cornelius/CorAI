import time

import numpy as np
import sklearn.model_selection
import torch

from priv_lib_ml.src.classes.estimator.history.estim_history import Estim_history
from priv_lib_ml.src.classes.training_stopper.early_stopper_vanilla import Early_stopper_vanilla
from priv_lib_ml.src.train.train import nn_train


def nn_kfold_train(data_train_X, data_train_Y, Model_NN, param_train,
                   early_stoppers=(Early_stopper_vanilla(),),
                   nb_split=5, shuffle_kfold=True, percent_val_for_1_fold=20,
                   hyper_param={}.copy(),
                   only_best_fold_history=False, silent=False):
    """
        Prepares the indices for kfold and calls multiplefold train.

    Args:
        data_train_X (tensor): Input data.
        data_train_Y (tensor): Target value.
        Model_NN (Class Savable_net): Parametrised architecture.
            Requirements: call constructor over it to create a net.
        param_train (NNTrainParameters): The parameters used for training.
        early_stoppers (iterable of Early_stopper): Used for deciding if the training should stop early.
            Preferably immutable to insure no changes.
        nb_split (int): The number of folds to split the data into.
        shuffle_kfold (bool): Flag to specify if the data should be shuffled.
        percent_val_for_1_fold (double): The percent of data that should be used for validation for the 1 fold case.
            Requirements: [0,100[
        hyper_param (dict): All the training parameters to be saved in the estimator.
        only_best_fold_history (bool): Flag to specify if only the history of the best fold should be saved.
        silent (bool): Verbose.

    Returns:
        best_net, estimator_history

    Post-condition :
        early_stoppers not changed.
    """
    # place where logs of trainings with respect to the metrics are stored.
    indices, compute_validation = _nn_kfold_indices_creation_random(data_train_X, data_train_Y,
                                                                    percent_val_for_1_fold, nb_split, shuffle_kfold)
    # Check if the correct type of early stoppers are passed
    if not compute_validation:
        for stop in early_stoppers:
            assert not stop.is_validation(), "Input validation stopper while no validation set given."

    # initialise estimator where history is stored.
    estimator_history = initialise_estimator(compute_validation, param_train, hyper_param)

    return _nn_multiplefold_train(data_train_X, data_train_Y, early_stoppers, Model_NN, nb_split, param_train, indices,
                                  silent, estimator_history, only_best_fold_history)


def initialise_estimator(compute_validation, param_train, train_param_dict={}):
    """
        Initialise the history estimator.
    Args:
        compute_validation (bool): Flag to specify is validation is included.
        param_train (NNTrainParameters): The parameters used for training.
        train_param_dict (dict): The hyper-parameters to be saved.

    Returns:
        Estim_history object.
    """
    metric_names = [metric.name for metric in param_train.metrics]
    # initialise the estimator history.
    estimator_history = Estim_history(metric_names=metric_names, validation=compute_validation,
                                      hyper_params=train_param_dict)
    return estimator_history


# section ######################################################################
#  #############################################################################
# MULTIFOLD

def _nn_multiplefold_train(data_train_X, data_train_Y,
                           early_stoppers, Model_NN, nb_split,
                           param_train, indices, silent,
                           estimator_history, only_best_fold_history=False):
    """
        Perform training over all the folds.
    Args:
        data_train_X (tensor): Input data.
        data_train_Y (tensor): Target value.
        early_stoppers (iterable of Early_stopper): Used for deciding if the training should stop early.
        Model_NN (Class Savable_net): Parametrised architecture.
            Requirements: call constructor over it to create a net.
        nb_split (int): The number of folds to split the data into.
        param_train (NNTrainParameters): The parameters used for training.
        indices (list of tuples): Each tuple contains the indices for training and for validation
        silent (bool): Verbose. Prints the nb of the best fold from 1 to K.
        estimator_history (Estim_history): An estimator object which stores the history of a training.
        only_best_fold_history (bool): Flag to specify if only the history of the best fold should be saved.

    Returns:
        The best_net, and the estimator_history
    """

    # for storing the network:
    value_metric_for_best_NN = - np.Inf  # : we set -\infty which can only be improved.
    # :Recall, the two criterea are either accuracy (so any accuracy is better than a neg. number)
    # : and minus loss, and a loss is always closer to zero than - infinity.
    best_net = None

    # : random_state is the seed of StratifiedKFold.
    for i, (index_training, index_validation) in enumerate(indices):
        if not silent:
            time.sleep(0.0001)  # for printing order
            print(f"{i + 1}-th Fold out of {nb_split} Folds.")
            time.sleep(0.0001)  # for printing order

        # : one can use tensors as they are convertible to numpy.
        (best_net, value_metric_for_best_NN) = train_kfold_a_fold_after_split(data_train_X, data_train_Y,
                                                                              index_training,
                                                                              index_validation, Model_NN, param_train,
                                                                              estimator_history, early_stoppers,
                                                                              value_metric_for_best_NN,
                                                                              best_net, i, silent)

    if not silent:
        print("Finished the K-Fold Training, the best NN is the number {}".format(estimator_history.best_fold + 1))

    if only_best_fold_history:
        estimator_history.slice_best_fold()

    return best_net, estimator_history


def train_kfold_a_fold_after_split(data_train_X, data_train_Y, index_training, index_validation, Model_NN, param_train,
                                   estimator_history, early_stoppers=(Early_stopper_vanilla(),),
                                   value_metric_for_best_NN=-np.Inf, best_net=None, i=0,
                                   silent=False):
    """
        Train one fold.
    Note:
        Can be used for training if the indices are already generated.
    Args:
        data_train_X (tensor): Input data.
        data_train_Y (tensor): Target data.
        index_training (slice): Indices used to slice the data for training.
        index_validation (slice): Indices used to slice the data for validation.
        Model_NN (Class Savable_net): Parametrised architecture.
            Requirements: call constructor over it to create a net.
        param_train (NNTrainParameters): The parameters used for training.
        estimator_history (Estim_history): The estimator in which the results will be saved.
        early_stoppers (iterable of Early_stopper): Used for deciding if the training should stop early.
        value_metric_for_best_NN: #todo Niels explain this
        best_net (Savable_net): Best net so far, based on comparison.
        i (int): Number of fold.
            Requirements: 0 <= i < nb_of_split
        silent (bool): Verbose.

    Returns:
        best_net, value_metric_for_best_net, number_kfold_best_net

    Post-conditions:
        estimator_history is updated to contain the training and the parameter best_fold updated.
        early stoppers are not modified.
        value_metric_for_best_NN is modified.
        best_net is modified for the new net.
        i is not modified.
    """
    net = Model_NN().to(param_train.device)

    # reset the early stoppers for the following fold
    for early_stopper in early_stoppers:
        early_stopper.reset()

    start_train_fold_time = time.time()
    kfold_history, kfold_best_epoch = nn_train(net, data_X=data_train_X, data_Y=data_train_Y,
                                               params_training=param_train, indic_train_X=index_training,
                                               indic_train_Y=index_training, early_stoppers=early_stoppers,
                                               indic_val_X=index_validation, indic_val_Y=index_validation,
                                               silent=silent)  # train network and save results
    end_train_fold_time = time.time() - start_train_fold_time
    history = _translate_history_to_dataframe(history=kfold_history,
                                              fold_number=i,
                                              validation=estimator_history.validation)
    estimator_history.append(history=history, fold_best_epoch=kfold_best_epoch, fold_time=end_train_fold_time)

    return _new_best_model(best_net, i, net, value_metric_for_best_NN, estimator_history, silent)


def _new_best_model(best_net, i, net, value_metric_for_best_NN, estimator_history, silent):
    """
        Compare the new results and save the best net.
    Args:
        best_net (Savable_net): Current best net.
        i: The index of the current fold.
        net: Current net.
        value_metric_for_best_NN: #todo Niels, find explanation
        estimator_history (Estim_history): The estimator in which the results will be saved.
        silent (bool): Verbose.

    Returns:
        best_net, value_metric_for_best_NN, number_kfold_best_net
    """
    rookie_perf = -estimator_history.get_values_fold_epoch_col(i, estimator_history.list_best_epoch[i], "loss_training")

    if not silent:  # -1 * ... bc we want to keep order below :
        print("New best model updated: rookie perf : {:e}"
              " and old best perf : {:e}.".format(-rookie_perf, -value_metric_for_best_NN))
    if value_metric_for_best_NN < rookie_perf:
        best_net = net
        value_metric_for_best_NN = rookie_perf
        estimator_history.best_fold = i
    return best_net, value_metric_for_best_NN


# section ######################################################################
#  #############################################################################
# INDICES

def _nn_kfold_indices_creation_random(data_training_X, data_training_Y,
                                      percent_valid_for_1_fold,
                                      nb_split, shuffle_kfold):
    """
        Computes the indices to be used to split the data into training and validation
    Args:
        data_training_X (tensor): Input data.
        data_training_Y (tensor): Target data.
        percent_valid_for_1_fold (double): The percent of data that should be used for validation for the 1 fold case.
            Requirements: [0,100[
        nb_split (int): The number of folds to split the data into.
        shuffle_kfold (bool): Flag to specify if the data should be shuffled.

    Returns:
        A list of tuples (a tuple per fold) and a bool = compute_validation
    """
    # Only one fold
    if nb_split == 1:
        assert 0 <= percent_valid_for_1_fold < 100, "percent_validation_for_1_fold should be in [0,100[ !"

        # Without validation fold
        if percent_valid_for_1_fold == 0:
            return [(torch.arange(data_training_X.shape[0]), None)], False
            # : kfold split hands back list of tuples. List container, a tuple for each fold.

        training_size = int((100. - percent_valid_for_1_fold) / 100. * data_training_X.shape[0])
        if shuffle_kfold:
            # for the permutation, one could look at https://discuss.pytorch.org/t/shuffling-a-tensor/25422/7:
            # we simplify the expression bc our tensors are in 2D only:
            indices = torch.randperm(data_training_X.shape[0])
            #: create a random permutation of the range( nb of data )

            indic_train = indices[:training_size]
            indic_validation = indices[training_size:]
        else:
            indic_train = torch.arange(training_size)
            indic_validation = torch.arange(training_size, data_training_X.shape[0])
        return [(indic_train, indic_validation)], True
        # : kfold split hands back list of tuples. List container, a tuple for each fold.

    # multiple folds
    else:
        try:
            # classification
            kfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold)
            # the seed is not fixed here but outside.

            # attempt to use the indices to check whether we can use stratified kfold
            for _ in kfold.split(data_training_X, data_training_Y):
                break

        except ValueError:
            # regression
            kfold = sklearn.model_selection.KFold(n_splits=nb_split, shuffle=shuffle_kfold)
            # the seed is not fixed here but outside.

        return kfold.split(data_training_X, data_training_Y), True


def _translate_history_to_dataframe(history, fold_number, validation):
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
        new_value = value[~np.isnan(value)] # we slice out the value that are nan.
        translated_history[new_key] = new_value.tolist()

    assert ('validation' in history) == validation, "The information about validation in estimator " \
                                                    "is not reflected in history"
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
