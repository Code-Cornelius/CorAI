import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from keras.datasets import mnist
from priv_lib_plot import APlot
from torch import nn

from priv_lib_ml.src.classes.estimator.history.relplot_history import Relplot_history
from priv_lib_ml.src.train.kfold_training import nn_kfold_train
from priv_lib_ml.src.nn_plots import confusion_matrix_creator
from priv_lib_ml.src.classes.architecture.fully_connected import factory_parametrised_FC_NN
from priv_lib_ml.src.classes.metric.metric import Metric
from priv_lib_ml.src.classes.optim_wrapper import Optim_wrapper
from priv_lib_ml.src.classes.training_stopper.early_stopper_training import Early_stopper_training
from priv_lib_ml.src.classes.training_stopper.early_stopper_validation import Early_stopper_validation
from priv_lib_ml.src.train.nntrainparameters import NNTrainParameters
from priv_lib_ml.src.util_training import set_seeds, pytorch_device_setting



# set seed for pytorch.
set_seeds(42)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 10000
device = pytorch_device_setting('not_cpu_please')
SILENT = False
early_stop_train = Early_stopper_training(patience=200, silent=SILENT, delta=-0.05)
early_stop_valid = Early_stopper_validation(patience=200, silent=SILENT, delta=-0.05)
early_stoppers = (early_stop_train, early_stop_valid)

accuracy_wrapper = lambda net, xx, yy: sklearn.metrics.accuracy_score(net.nn_predict_ans2cpu(xx),
                                                                      yy.reshape(-1, 1).to('cpu'),
                                                                      normalize=False
                                                                      )
accuracy_metric = Metric(name="accuracy", function=accuracy_wrapper)
metrics = (accuracy_metric,)
#############################
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
train_Y = pd.DataFrame(train_y)

test_X = pd.DataFrame(test_X.reshape(10000, 28 * 28))
test_Y = pd.DataFrame(test_y)

train_X = train_X[:n_samples]
train_Y = train_Y[:n_samples]
test_X = test_X[:n_samples]
test_Y = test_Y[:n_samples]

train_X = torch.from_numpy(train_X.values).float()
train_Y = torch.from_numpy(train_Y.values).long().squeeze()  # squeeze for compatibility with loss function
test_X = torch.from_numpy(test_X.values).float()
test_Y = torch.from_numpy(test_Y.values).long().squeeze()  # squeeze for compatibility with loss function

if __name__ == '__main__':
    # config of the architecture:
    input_size = 28 * 28
    hidden_sizes = [100]
    output_size = 10
    biases = [True, True]
    activation_functions = [F.relu]
    dropout = 0.2
    epochs = 1000
    batch_size = 1000
    optimiser = torch.optim.SGD
    criterion = nn.CrossEntropyLoss(reduction='sum')
    dict_optimiser = {"lr": 0.0000005, "weight_decay": 0.00001}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)

    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)
    Class_Parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                       param_list_hidden_sizes=hidden_sizes,
                                                       param_output_size=output_size, param_list_biases=biases,
                                                       param_activation_functions=activation_functions,
                                                       param_dropout=dropout,
                                                       param_predict_fct=lambda out: torch.max(out, 1)[1])

    (net, estimator_history) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN, param_train=param_training,
                                              early_stoppers=early_stoppers, nb_split=1, shuffle_kfold=True,
                                              percent_val_for_1_fold=10, silent=False)

    # fetch the best value and assert if accuracy > threshold.

    history_plot = Relplot_history(estimator_history)
    history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot='accuracy', log_axis_for_loss=True,
                                            log_axis_for_second_axis=False)

    history_plot.lineplot(log_axis_for_loss=True)

    net.to('cpu')
    prediction = net.nn_predict(train_X)
    # estimator_history.err_compute_best_net(net, train_X, train_Y, testing_X=test_X, testing_Y=test_Y, device='cpu')
    confusion_matrix_creator(train_Y, prediction, range(10), title="")

    APlot.show_plot()
