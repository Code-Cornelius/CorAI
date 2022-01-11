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
    """ the func needs to have net and device in its definition.
    The decorator adds the parameters if the original functions does not have them, which could be error prone."""

    @functools.wraps(func)
    def wrapper_decorator_on_cpu_during_fct(*, net, device='cpu', **kwargs):
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
    If 'cpu' given, sets cpu
    else, see if cuda available, otherwise cpu.

    Args:
        type (str):
        silent (bool):

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


def create_model_by_index(index, path2json, path2net,
                          config_architecture, mapping_names2functions,
                          flag_factory=False, **kwargs):
    """
        Create a model using the index when the parameters of the model can be found in a json.
    Args:
        index(int): The index of the parameters used for training in the json file.
        path2json(str): The path to the json file containing the training parameters (as list of dicts).
        path2net(str): The path to the saved model, where the trained parameters are stored.
        config_architecture(callable): A callable returning the class (or the instance, depending on flag_factory)
            that will be used to initialise the model. It is able to create the model
            if given the parameters fetch from the json.
        mapping_names2functions(dict): A mapping from the string name of a function to the python function. Check
            the function `replace_function_names_to_functions`.
        flag_factory(bool): if true, config_architecture returns a class and is instantiated with ().
            Otherwise, returns an instance and is not re-instantiated.

    Returns:
        The nn model.

    Examples:
        see the tutorial: tutorial_estim_hyperparameter.md .
    """
    dict_params = retrieve_parameters_by_index_from_json(index, path2json)
    replace_function_names_to_functions(dict_params, mapping_names2functions, silent=True)

    print(f"For config {index}, the parameters are : \n{dict_params}.")
    if flag_factory:
        parametrized_NN = config_architecture(dict_params, **kwargs)().load_net(path2net)
    else:
        parametrized_NN = config_architecture(dict_params, **kwargs).load_net(path2net)
    return parametrized_NN

############### old code for create_model_bu_index:
# assert set(keys_to_pop).issubset(
#     set(list_of_names_args)), "list_of_names_args must contain all elements of keys_to_pop."

# if list_of_names_args is None: # no need to filter and rename keywords
#     dict_params = parameters
# else :
#     dict_params = {key: value for key, value in
#                zip(list_of_names_args, list(parameters.values())
#                    )}
# for key in keys_to_pop:
#     dict_params.pop(key)

#         list_of_names_args(list of str): order should be the same as the dict_params' values, read from path2json.
#         keys_to_pop(list of str): the list should be a subset of list_of_names_args.
#                                   Keys removed from the fetched parameters.
#                                   The one that should not be given to the model.