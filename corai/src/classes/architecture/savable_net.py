# global libraries
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn

from corai.src.util_train import decorator_train_disable_no_grad
# my lib
from corai_error import Error_type_setter


class Savable_net(nn.Module):
    """
    Semantics:
        A savable net defines all the required members in order to save the best net,
        and use a net with training_stoppers.
    Args:
        best_weights:
        best_epoch:
    Class Args:
        _predict_fct
    """

    # function that from the output returns the prediction. Depends on the problem:
    _predict_fct = nn.Identity()

    # : default predict_fct. Can be masked with lower child class functions.
    # : the hidden mark "_" is important to not pass through the setter but directly to the parameter.
    # : we set the class variable, that is also defined as an object variable unless redefined!

    def __init__(self, predict_fct, *args, **kwargs):
        """
        Constructor for Neural Network.
        """
        super().__init__()
        self.predict_fct = predict_fct  # put None if you want to keep default
        # best parameters, keeps track in case of early stopping.
        self.best_weights = None  # init the field best weights.
        self.best_epoch = 0

    @property
    def device(self):
        # device of the model:
        # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        try:
            return next(self.parameters()).device
        except StopIteration:
            print("\n From Savable Net: no next parameter to determine the device. Program exits with error code 1.")
            sys.exit(1)

    def prediction(self, out):
        """returns the class predicted for each element of the tensor."""
        # gets the class that is max probability
        return self.predict_fct(out)

    def init_weights_of_model(self):
        """Initialise weights of the model such that they have a predefined structure"""
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        """gets called in init_weights_of_model"""
        # if linear, if the weights needs to be changed as well as bias.
        if type(layer) == nn.Linear and layer.weight.requires_grad and layer.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
            layer.bias.data.fill_(0)

    def save_net(self, path):
        """
        Create a directory if the net is saved in a place where the directory does not exist.
        Good practice is to use the extension .pt.
        """
        directory_where_to_save = os.path.dirname(path)
        if not os.path.exists(directory_where_to_save):
            os.makedirs(directory_where_to_save)
        torch.save(self.state_dict(), path)
        return self

    def load_net(self, path):
        self.load_state_dict(torch.load(path))
        return self

    ######
    # Something like this is nice:
    # loaded_model = cls()
    # res = loaded_model.load_state_dict(torch.load(PATH_TO_MODEL))
    # if res.missing_keys:
    #     raise AttributeError(f"Missing keys: {res.missing_keys}")
    # return loaded_model

    def update_best_weights(self, epoch):
        # : We decide to keep a copy instead of saving the model in a file
        # because we might not want to save this model (E.G. if we do a K-FOLD)
        self.best_weights = deepcopy(self.state_dict())  # used in early stoppers.
        self.best_epoch = epoch

    # section ######################################################################
    #  #############################################################################
    # prediction

    @decorator_train_disable_no_grad
    def nn_predict(self, data_to_predict):
        """
        Semantics : pass data_to_predict through the neural network and returns its prediction.
        The output data is going through the net.prediction() function.
        Also, we request the device, where the input, the net, and output lies.

        Condition: net has the method prediction.

        Args:
            data_to_predict:

        Returns:

        """
        # ~~~~~~~~~~~~~~~~~~ to device for optimal speed, though we take the data back with .cpu().
        # we do not put the data on GPU! As the overhead might be too much.
        data_predicted = self.prediction(self(data_to_predict))  # forward pass
        return data_predicted

    def nn_predict_ans2cpu(self, data_to_predict):
        return self.nn_predict(data_to_predict).cpu()

    # section ######################################################################
    #  #############################################################################
    # SETTER GETTER

    @property
    def predict_fct(self):
        # function that from the output returns the prediction. Depends on the problem:
        return self._predict_fct

    @predict_fct.setter
    def predict_fct(self, new_predict_fct):
        if new_predict_fct is None:
            pass
        else:
            if callable(new_predict_fct):
                self._predict_fct = new_predict_fct
            else:
                raise Error_type_setter(f"Argument is not callable.")

    @property
    def best_weights(self):
        return self._best_weights

    @best_weights.setter
    def best_weights(self, new_best_weights):
        self._best_weights = new_best_weights
