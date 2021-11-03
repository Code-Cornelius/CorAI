import numpy as np
import pandas as pd
import torch
from corai_plot import APlot
from corai_util.tools.src.benchmarking import benchmark
from torch import nn
from tqdm import tqdm

from corai.src.classes.estimator.estim_benchmark_perf_nn_sizes import Estim_benchmark_perf_nn_sizes, \
    Relplot_benchmark_perf_nn_sizes
from corai.src.train.kfold_training import nn_kfold_train
from corai.src.classes.architecture.fully_connected import factory_parametrised_FC_NN
from corai.src.classes.optim_wrapper import Optim_wrapper
from corai.src.train.nntrainparameters import NNTrainParameters
from corai.src.util_train import set_seeds, pytorch_device_setting

# set seed for pytorch.
set_seeds(42)

READ_FROM_SAVED = True


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


# 80% is taken for training
# batch is 1/4 of total training size.
sizes_samples = [50, 400, 1600, 3600, 6400,
                 12800, 25600, 51200, 100000, 200000, 500000]
# only divisble by 5 number in order to be ok for splitting sizes.

sizes_model = [5, 20, 160, 640, 2048, 4096, 8192, 16384, 32768]

depths = [1]
processing_units = ['gpu', 'cpu']

epochs = 40


def benchmark_and_save(estim, input_size, type_pu, size_model, depth, **kwargs):
    time = benchmark(nn_kfold_train, number_of_rep=1, silent_benchmark=True, **kwargs) / epochs
    time_dict = {"Input Size": input_size,
                 "Processing Unit": type_pu,
                 "Model Size": size_model,
                 'Depth': depth,
                 "Comput. Time": [time]}
    estim.append(pd.DataFrame(time_dict))
    return


if not READ_FROM_SAVED:
    estim_bench = Estim_benchmark_perf_nn_sizes(epochs)

    for size_sample in tqdm(sizes_samples):
        for PU in processing_units:
            for size_model in sizes_model:
                for depth in depths:
                    if PU == 'cpu' and size_sample > 12800:
                        break  # too long
                    try:  # escape for too big models in gpu
                        sigma = 0.0  # Noise level
                        device = pytorch_device_setting(PU, True)
                        SILENT = False
                        xx = 2 * np.pi * torch.rand((size_sample, 1))
                        yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

                        training_size = int(80. / 100. * size_sample)
                        train_X = xx[:training_size, :]
                        train_Y = yy[:training_size, :]

                        input_size = 1
                        hidden_sizes = [size_model] * depth
                        output_size = 1
                        biases = [True] * depth + [True]
                        activation_functions = [torch.relu] * depth
                        dropout = 0.
                        batch_size = training_size // 4
                        optimiser = torch.optim.Adam
                        criterion = nn.MSELoss(reduction='sum')
                        dict_optimiser = {"lr": 0.0005, "weight_decay": 1E-7}
                        optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)
                        param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                                           criterion=criterion, optim_wrapper=optim_wrapper)
                        parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                                     param_list_hidden_sizes=hidden_sizes,
                                                                     param_output_size=output_size,
                                                                     param_list_biases=biases,
                                                                     param_activation_functions=activation_functions,
                                                                     param_dropout=dropout,
                                                                     param_predict_fct=None)
                        benchmark_and_save(estim_bench, size_sample, PU, size_model, depth,
                                           data_train_X=train_X, data_train_Y=train_Y,
                                           Model_NN=parametrized_NN, param_train=param_training,
                                           nb_split=1, shuffle_kfold=False,
                                           percent_val_for_1_fold=0, silent=True)
                    except RuntimeError as e:
                        print(e)
                        print(f"error for model {size_model}, sample {size_sample}, under {PU}.")
                        break
    estim_bench.to_csv("simple_benchmark_cpu_gpu_size_input.csv")
else:
    estim_bench = Estim_benchmark_perf_nn_sizes.from_csv("simple_benchmark_cpu_gpu_size_input.csv")

plot_evol_estim = Relplot_benchmark_perf_nn_sizes(estim_bench)
plot_evol_estim.lineplot(column_name_draw='Comput. Time', envelope_flag=False,
                         separators_plot=["Processing Unit"], hue="Model Size",
                         dict_plot_for_main_line={}, markers=True, style="Model Size")
APlot.show_plot()
