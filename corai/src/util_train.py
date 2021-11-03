import functools

import numpy as np
import torch
import torch.cuda
from corai_util.tools.src.function_dict import \
    retrieve_parameters_by_index_from_json, \
    replace_function_names_to_functions


def decorator_train_disable_no_grad(func):
    """
    Args:
        func:

    Returns:

    """

    @functools.wraps(func)
    def wrapper_decorator_train_disable_no_grad(net, *args, **kwargs):
        net.train(mode=False)  # Disable dropout and normalisation
        with torch.no_grad():  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
            ans = func(net, *args, **kwargs)
        net.train(mode=True)  # Re-able dropout and normalisation
        return ans

    return wrapper_decorator_train_disable_no_grad


def decorator_on_cpu_during_fct(func):
    """ the func needs to have net and device in its definition."""

    @functools.wraps(func)
    def wrapper_decorator_on_cpu_during_fct(*, net, device, **kwargs):
        # key words only.
        net.to(torch.device('cpu'))
        ans = func(net=net, **kwargs)
        net.to(device)
        return ans

    return wrapper_decorator_on_cpu_during_fct


def pytorch_device_setting(type='', silent=False):
    """
    Semantics : sets the device for NeuralNetwork computations.
    Put nothing for automatic choice.
    If cpu given, sets cpu
    else, see if cuda available, otherwise cpu.

    Args:
        type:

    Returns:

    """
    device = torch.device('cpu') if type == 'cpu' else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not silent:
        print("Script running on device : ", device)
    return device


def set_seeds(seed):
    """
    Semantics:
        Set the seed of torch and numpy.
    Args:
        seed(int): The new seed.

    Returns:
        Void.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_model_by_index(index, path2json,
                          path2net, Model_nn, mapping_names2functions,
                          list_of_names_args, *args, **kwargs):
    """
        Create a model using the index when the parameters of the model can be found in a json.
    Args:
        index(int): The index of the parameters used for training in the json file.
        path2json(str): The path to the json file containing the training parameters.
        path2net(str): The path to the saved model.
        Model_nn(class): The class that will be used to initialise the model.
        mapping_names2functions(dict): A mapping from the string name of a function to the python function.
        list_of_names_args(list of str):

    Returns:
        The nn model.
    """
    parameters = retrieve_parameters_by_index_from_json(index, path2json)
    print(f"For config {index}, the parameters are : \n{parameters}.")
    replace_function_names_to_functions(parameters, mapping_names2functions, silent=True)
    dict_params = {key: value
                   for key, value in zip(list_of_names_args,
                                         list(parameters.values())
                                         )
                   }
    parametrized_NN = Model_nn(*args, **dict_params, **kwargs)().load_net(path2net)
    return parametrized_NN
