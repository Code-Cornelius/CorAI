import os

import numpy as np
import torch
from torch import linalg as LA
from torch import nn
from tqdm import tqdm

import corai
from corai_plot import APlot
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
device = corai.pytorch_device_setting('cpu')
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
    "seed": [42, 124],
    "lr": [0.0001, 0.001, 0.01, 0.1],
    'activation_function': ['tanh', 'relu', 'celu'],
    "dropout": [0., 0.2, 0.5],
    "list_hidden_sizes": [[2, 4, 2], [4, 8, 4], [16, 32, 16], [2, 32, 2], [32, 128, 32]],
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


ROOTPATH = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(ROOTPATH, "example_hyper_param_sin_estim_history")
NEW_DATASET = False
SAVE_TO_FILE = True


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
        if SAVE_TO_FILE:
            estimator_history.to_json(path=os.path.join(FOLDER_PATH, f"estim_{i}.json"), compress=False)

    return estims


if __name__ == '__main__':

    if NEW_DATASET:
        estims = generate_estims_history()
    if not NEW_DATASET:
        estim_hyper_param = corai.Estim_hyper_param.from_folder(FOLDER_PATH,
                                                                metric_names=["loss_validation", "loss_training"],
                                                                flg_time=True, compressed=False)

        ######## example of usage:
        # estim_hyper_param = corai.Estim_hyper_param.from_list(estims, metric_names=["loss_validation", "loss_training"],
        #                                                 flg_time=True)
        # estim_hyper_param.to_csv("example_estim_hyper_param.csv")
        # estim_hyper_param.compute_number_params_for_fcnn()

        ######## drawing the distribution plot:
        histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
        histplot_hyperparam.hist(column_name_draw='train_time',  hue='dropout',
                                 separators_plot=None,
                                 palette='RdYlBu', bins=50,
                                 binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)
        histplot_hyperparam.hist(column_name_draw='loss_validation',
                                 separators_plot=None,
                                 hue='lr',
                                 palette='RdYlBu', bins=20,
                                 binrange=None, stat='count', multiple="dodge", kde=True, path_save_plot=None)

        ######## drawing the relation plot:
        scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
        scatplot_hyperparam.scatter(column_name_draw='loss_training',
                                    second_column_to_draw_abscissa='loss_validation',
                                    hue='train_time',
                                    hue_norm=(0, 30), legend=False)

        ######## conditioning the data to plot a subset:
        condition = lambda t: t <= 2
        estim_hyper_param.slice(column='train_time', condition=condition, save=True)  # slice data, removing some part.

        histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
        histplot_hyperparam.hist(column_name_draw='train_time',
                                 separators_plot=None,
                                 hue='dropout',
                                 palette='RdYlBu', bins=50,
                                 binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)

        histplot_hyperparam.hist(column_name_draw='loss_validation',
                                 separators_plot=None,
                                 hue='lr',
                                 palette='RdYlBu', bins=20,
                                 binrange=None, stat='count', multiple="dodge", kde=True, path_save_plot=None)

        scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
        scatplot_hyperparam.scatter(column_name_draw='loss_training', second_column_to_draw_abscissa='loss_validation',
                                    hue='train_time', hue_norm=(0, 30), legend=False)
        APlot.show_plot()
