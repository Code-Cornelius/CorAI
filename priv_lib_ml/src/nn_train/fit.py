from matplotlib import pyplot as plt
from priv_lib_util.tools import function_iterable
from tqdm import tqdm

from src.nn_classes.fast_tensor_dataloader import FastTensorDataLoader
from src.nn_classes.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla
from src.util_training import decorator_train_disable_no_grad

PLOT_WHILE_TRAIN = False
FREQ_NEW_IMAGE = 40


def plot_while_training(params_training, history, ax):
    ax.clear()
    ax.semilogy(range(params_training.epochs), history['training']['loss'], 'b', label='Train Loss')
    ax.semilogy(range(params_training.epochs), history['validation']['loss'], 'r', label='Validation Loss')
    ax.legend(loc="best")
    plt.pause(0.0001)


def nn_fit(net, X_train_on_device, Y_train_on_device,
           params_training, history,
           early_stoppers=(Early_stopper_vanilla(),),
           X_val_on_device=None, Y_val_on_device=None,
           *, silent=False):
    """
    Args:
        net (Savable_net): model.
        X_train_on_device:
        Y_train_on_device:
        params_training: type NNTrainParameters, the parameters used in training
        history: collection of results from metrics
        early_stoppers: iterable of Early_stopper. Used for deciding if the training should stop early.
            Preferably immutable to insure no changes.
        X_val_on_device:
        Y_val_on_device:
        silent: verbose.

    Returns: epoch of the best net and updates the history

    Post-condition :
        early_stoppers not changed.
    """
    if PLOT_WHILE_TRAIN:
        fig = plt.figure('plot while training')
        ax = fig.add_subplot(111)

    # condition if we use validation set.
    (criterion, is_validat_included, total_number_data,
     train_loader_on_device, validat_loader_on_device) = prepare_data_for_fit(X_train_on_device, X_val_on_device,
                                                                              Y_train_on_device, Y_val_on_device, net,
                                                                              params_training)

    epoch = 0
    for epoch in tqdm(range(params_training.epochs), disable=silent):  # disable unable the print.
        ###################
        # train the model #
        ###################
        train_loss = 0  #: aggregate variable
        for i, (batch_X, batch_y) in enumerate(train_loader_on_device, 0):
            # closure needed for some algorithm.
            def closure():
                # The closure should clear the gradients, compute the loss, and return it.
                # https://pytorch.org/docs/stable/optim.html

                # set gradients to zero
                params_training.optim_wrapper.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                # Do forward and backward pass
                loss = criterion(net(batch_X), batch_y)  #: compute the loss : difference of result and expectation
                loss.backward()  # : compute the gradients
                return loss

            # Optimisation step
            params_training.optim_wrapper(closure=closure)  # : update the weights

            # you need to call again criterion, as we cannot store the criterion result:
            train_loss += criterion(net(batch_X), batch_y).item()
            #: weight the loss accordingly. That is the reason why using average is flawed.

        # adjust the learning rate if a scheduler is used, must be called after optimiser.step was called
        params_training.optim_wrapper.update_learning_rate()

        # Normalize and save the loss over the current epoch:
        history['training']['loss'][epoch] = train_loss / total_number_data[0]
        _update_history(net, params_training.metrics, criterion, epoch, is_validat_included, total_number_data,
                        train_loader_on_device, validat_loader_on_device, history)

        ######################
        #   Early Stopping   #
        ######################
        # check if any improvement and update early.has_improved_last_epoch.
        if _do_early_stop(net, early_stoppers, history, epoch, silent):
            # does not break first epoch_iter because
            # _early_stopped = False,
            # has_improved_last_epoch = True
            break # stop epoch

        # Check if NN has not improved with respect to the early stoppers.
        # If has not, we do not improve the best_weights of the NN
        if all(early_stopper.has_improved_last_epoch for early_stopper in early_stoppers):
            net.update_best_weights(epoch)


        if PLOT_WHILE_TRAIN:
            if epoch % FREQ_NEW_IMAGE == 0:
                plot_while_training(params_training, history, ax)

    # ~~~~~~~~ end of the for in epoch. Training
    return _return_the_stop(net, epoch, early_stoppers)


def _do_early_stop(net, early_stoppers, history, epoch, silent):
    for early_stopper in early_stoppers:
        if early_stopper(net, history, epoch): # if any earlly stops, then we return true.
            if not silent: print("Terminated epochs, with early stopper training at epoch {}.".format(epoch))
            return True
    return False


def prepare_data_for_fit(X_train_on_device, X_val_on_device, Y_train_on_device, Y_val_on_device, net, params_training):
    list_params_validat = [X_val_on_device, Y_val_on_device]

    is_validat_included = not function_iterable.are_at_least_one_None(list_params_validat)
    #: equivalent to are all not None ?
    # raise if there is a logic error.
    if is_validat_included:  #: if we need validation
        total_number_data = Y_train_on_device.shape[0], Y_val_on_device.shape[0]  # : constants for normalisation
        # create data validat_loader : load validation data in batches
        validat_loader_on_device = FastTensorDataLoader(X_val_on_device, Y_val_on_device,
                                                        batch_size=params_training.batch_size,
                                                        shuffle=False)  # SHUFFLE IS COSTLY!
    else:
        total_number_data = Y_train_on_device.shape[0], 0  # : constants for normalisation
        function_iterable.raise_if_not_all_None(list_params_validat)
        validat_loader_on_device = None  # in order to avoid referenced before assigment
    # create data train_loader_on_device : load training data in batches
    train_loader_on_device = FastTensorDataLoader(X_train_on_device, Y_train_on_device,
                                                  batch_size=params_training.batch_size, shuffle=True)
    # : SHUFFLE IS COSTLY! it is the only shuffle really useful

    # pick loss function and optimizer
    criterion = params_training.criterion
    params_training.optim_wrapper.initialise_optimiser(net.parameters())
    return criterion, is_validat_included, total_number_data, train_loader_on_device, validat_loader_on_device


def _update_history(net, metrics, criterion, epoch, is_valid_included, total_number_data, train_loader_on_device,
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
        _update_validation_loss(net, criterion, epoch, total_number_data, history, validat_loader_on_device)

        #######################
        # Validation Metrics  #
        #######################
        for metric in metrics:
            _update_history_for_metric(metric, net, epoch, total_number_data, history, validat_loader_on_device,
                                       'validation')

    return


def _update_history_for_metric(metric, net, epoch, total_number_data, history, data_loader, type):
    history[type][metric.name][epoch] = 0
    for batch_X, batch_y in data_loader:
        history[type][metric.name][epoch] += metric(net, batch_X, batch_y)

    history[type][metric.name][epoch] /= total_number_data[0] if type == 'training' else total_number_data[1]


@decorator_train_disable_no_grad  # make sure we don't back propagate any loss over this data
def _update_validation_loss(net, criterion, epoch, total_number_data, history, validat_loader_on_device):
    history['validation']['loss'][epoch] = 0  # :aggregate variable
    for batch_X, batch_y in validat_loader_on_device:
        history['validation']['loss'][epoch] += criterion(net(batch_X), batch_y).item()
    history['validation']['loss'][epoch] /= total_number_data[1]


def _return_the_stop(net, current_epoch, early_stoppers):
    # args should be early_stoppers (or none if not defined)
    # multiple early_stoppers can't break at the same time,
    # because there will be a first that breaks out the loop first.
    # if no early_stopper broke, return the current epoch.
    for stopper in early_stoppers:
        if stopper.is_stopped():  #: check if the stopper is none or actually of type early stop.
            (net.load_state_dict(net.best_weights))  # .to(device)
            return net.best_epoch
    return current_epoch
