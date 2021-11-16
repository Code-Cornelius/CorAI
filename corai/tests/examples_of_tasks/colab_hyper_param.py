import os
import sys

import numpy as np
import torch
from torch import linalg as LA
from torch import nn
from tqdm import tqdm

import corai
from corai_util.tools import function_dict


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


def L4loss(net, xx, yy):
    return torch.norm(net.nn_predict(xx) - yy, 4)


L4metric = corai.Metric('L4', L4loss)
metrics = (L4metric,)

############################## GLOBAL PARAMETERS
n_samples = 2000  # Number of training samples
sigma = 0.01  # Noise level
device = corai.pytorch_device_setting()
SILENT = False
early_stop_train = corai.Early_stopper_training(patience=20, silent=SILENT, delta=-int(1E-6))
early_stop_valid = corai.Early_stopper_validation(patience=20, silent=SILENT, delta=-int(1E-6))
early_stoppers = (early_stop_train, early_stop_valid)
############################# DATA CREATION
# exact grid
plot_xx = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
plot_yy = exact_solution(plot_xx).reshape(-1, )
plot_yy_noisy = (exact_solution(plot_xx) + sigma * torch.randn(plot_xx.shape)).reshape(-1, )

# random points for training
xx = 2 * np.pi * torch.rand((n_samples, 1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

# slicing:
training_size = int(90. / 100. * n_samples)
train_X = xx[:training_size, :]
train_Y = yy[:training_size, :]

testing_X = xx[training_size:, :]
testing_Y = yy[training_size:, :]

##### end data

params_options = {
    "architecture": ["fcnn"],
    "seed": [42],
    "lr": [0.0001, 0.001],
    'activation_function': ['tanh', 'relu'],
    "dropout": [0., 0.2],
    "list_hidden_sizes": [[2, 4, 2], [4, 8, 4]],
}

hyper_params = function_dict.parameter_product(params_options)


def config_architecture(params):
    # config of the architecture:
    input_size = 1
    hidden_sizes = params["list_hidden_sizes"]
    output_size = 1
    biases = [True, True, True, True]
    if params['activation_function'] == 'tanh':
        activation_functions = [torch.tanh, torch.tanh, torch.tanh]
    elif params['activation_function'] == 'celu':
        activation_functions = [torch.celu, torch.celu, torch.celu]
    else:
        activation_functions = [torch.relu, torch.relu, torch.relu]

    dropout = params["dropout"]
    epochs = 7500
    batch_size = 1000
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": params["lr"], "weight_decay": 0.0000001}
    optim_wrapper = corai.Optim_wrapper(optimiser, dict_optimiser)
    param_training = corai.NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                             criterion=criterion, optim_wrapper=optim_wrapper,
                                             metrics=metrics)
    Class_Parametrized_NN = corai.factory_parametrised_FC_NN(param_input_size=input_size,
                                                             param_list_hidden_sizes=hidden_sizes,
                                                             param_output_size=output_size, param_list_biases=biases,
                                                             param_activation_functions=activation_functions,
                                                             param_dropout=dropout,
                                                             param_predict_fct=None)

    return param_training, Class_Parametrized_NN


def generate_estims_history():
    estims = []
    for i, params in enumerate(tqdm(hyper_params)):
        # set seed for pytorch.
        corai.set_seeds(params["seed"])

        param_training, Class_Parametrized_NN = config_architecture(params)

        (net, estimator_history) = corai.nn_kfold_train(train_X, train_Y,
                                                        Class_Parametrized_NN,
                                                        param_train=param_training,
                                                        early_stoppers=early_stoppers,
                                                        nb_split=1, shuffle_kfold=True,
                                                        percent_val_for_1_fold=20,
                                                        silent=True, hyper_param=params)
        estims.append(estimator_history)

    return estims


if __name__ == '__main__':
    estims = generate_estims_history()
    estim_hyper_param = corai.Estim_hyper_param.from_list(estims,
                                                          metric_names=["loss_validation", "loss_training"],
                                                          flg_time=True)
    data_path = sys.argv[1]
    estim_hyper_param.to_csv(data_path)
