from unittest import TestCase

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch import nn

from config import ROOT_DIR
from corai.src.classes.architecture.fully_connected import factory_parametrised_FC_NN
from corai.src.classes.metric.metric import Metric
from corai.src.classes.optim_wrapper import Optim_wrapper
from corai.src.classes.training_stopper.early_stopper_training import Early_stopper_training
from corai.src.classes.training_stopper.early_stopper_validation import Early_stopper_validation
from corai.src.train.kfold_training import nn_kfold_train
from corai.src.train.nntrainparameters import NNTrainParameters
from corai.src.util_train import set_seeds, pytorch_device_setting
from corai_util.tools.src.function_writer import factory_fct_linked_path


class Test_classification(TestCase):
    def setUp(self) -> None:
        # set seed for pytorch.
        set_seeds(42)

        ############################## GLOBAL PARAMETERS
        # Number of training samples
        n_samples = 10000
        device = pytorch_device_setting('not_cpu_please')
        SILENT = False
        early_stop_train = Early_stopper_training(patience=200, silent=SILENT, delta=-0.05)
        early_stop_valid = Early_stopper_validation(patience=200, silent=SILENT, delta=-0.05)
        self.early_stoppers = (early_stop_train, early_stop_valid)

        accuracy_wrapper = lambda net, xx, yy: sklearn.metrics.accuracy_score(net.nn_predict_ans2cpu(xx),
                                                                              yy.reshape(-1, 1).to('cpu'),
                                                                              normalize=False)
        accuracy_metric = Metric(name="accuracy", function=accuracy_wrapper)
        metrics = (accuracy_metric,)

        linker = factory_fct_linked_path(ROOT_DIR, "corai/tests/mnist_dataset")
        train_X = np.load(linker(["x_train.npy"]))
        train_y = np.load(linker(["y_train.npy"]))
        test_X = np.load(linker(["x_test.npy"]))
        test_y = np.load(linker(["y_test.npy"]))
        # (train_X, train_y), (test_X, test_y) = mnist.load_data() # instead we have saved the data
        train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
        train_Y = pd.DataFrame(train_y)

        train_X = train_X[:n_samples]
        train_Y = train_Y[:n_samples]

        self.train_X = torch.from_numpy(train_X.values).float()
        self.train_Y = torch.from_numpy(train_Y.values).long().squeeze()  # squeeze for compatibility with loss function

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

        self.param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                                criterion=criterion, optim_wrapper=optim_wrapper,
                                                metrics=metrics)
        self.Class_Parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                                param_list_hidden_sizes=hidden_sizes,
                                                                param_output_size=output_size, param_list_biases=biases,
                                                                param_activation_functions=activation_functions,
                                                                param_dropout=dropout,
                                                                param_predict_fct=lambda out: torch.max(out, 1)[1])

    def test_training(self):
        (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                  param_train=self.param_training,
                                                  early_stoppers=self.early_stoppers, nb_split=1, shuffle_kfold=True,
                                                  percent_val_for_1_fold=10, silent=True)

        best_acc = estimator_history.get_best_value_for('accuracy_validation')
        # : fetch the best value and assert if accuracy > threshold.
        assert best_acc > 0.7

    def test_training_kfold(self):
        (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                  param_train=self.param_training,
                                                  early_stoppers=self.early_stoppers, nb_split=5, shuffle_kfold=True,
                                                  percent_val_for_1_fold=10, silent=True)

        best_acc = estimator_history.get_best_value_for(
            'accuracy_validation')  # : fetch the best value and assert if accuracy > threshold.
        assert best_acc > 0.7


class Test_regression(TestCase):
    def setUp(self) -> None:
        # set seed for pytorch.
        set_seeds(42)

        # Define the exact solution
        def exact_solution(x):
            return torch.sin(x)

        ############################## GLOBAL PARAMETERS
        n_samples = 2000  # Number of training samples
        sigma = 0.01  # Noise level
        device = pytorch_device_setting('cpu')
        SILENT = False
        early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=-1E-6)
        early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=-1E-6)
        self.early_stoppers = (early_stop_train, early_stop_valid)
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
        self.train_X = xx[:training_size, :]
        self.train_Y = yy[:training_size, :]

        ##### end data

        def L4loss(net, xx, yy):
            return torch.norm(net.nn_predict(xx) - yy, 4)

        L4metric = Metric('L4', L4loss)
        metrics = (L4metric,)
        # config of the architecture:
        input_size = 1
        hidden_sizes = [20, 50, 20]
        output_size = 1
        biases = [True, True, True, True]
        activation_functions = [torch.tanh, torch.tanh, torch.relu]
        dropout = 0.
        epochs = 1000
        batch_size = 200
        optimiser = torch.optim.Adam
        criterion = nn.MSELoss(reduction='sum')
        dict_optimiser = {"lr": 0.001, "weight_decay": 0.0000001}
        optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)
        self.param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                                criterion=criterion, optim_wrapper=optim_wrapper,
                                                metrics=metrics)
        self.Class_Parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                                param_list_hidden_sizes=hidden_sizes,
                                                                param_output_size=output_size, param_list_biases=biases,
                                                                param_activation_functions=activation_functions,
                                                                param_dropout=dropout,
                                                                param_predict_fct=None)

    def test_training(self):
        (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                  param_train=self.param_training,
                                                  early_stoppers=self.early_stoppers, nb_split=1, shuffle_kfold=True,
                                                  percent_val_for_1_fold=20, silent=True)

        best_loss = estimator_history.get_best_value_for('loss_validation')
        # : fetch the best value and assert if accuracy > threshold.
        assert best_loss < 0.01

    def test_training_kfold(self):
        (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                  param_train=self.param_training,
                                                  early_stoppers=self.early_stoppers, nb_split=10, shuffle_kfold=True,
                                                  percent_val_for_1_fold=10, silent=True)

        best_loss = estimator_history.get_best_value_for('loss_validation')
        # : fetch the best value and assert if accuracy > threshold.
        assert best_loss < 0.001

    def test_training_no_val(self):
        try:
            (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                      param_train=self.param_training,
                                                      early_stoppers=self.early_stoppers, nb_split=1,
                                                      shuffle_kfold=True,
                                                      percent_val_for_1_fold=0, silent=True)
            raise PermissionError('Should have broke there, and the error caught! The error should have been: '
                                  'AssertionError: Input validation stopper while no validation set given.')


        except AssertionError:
            early_stoppers = (Early_stopper_training(patience=20, silent=True, delta=-1E-6),)

            (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                      param_train=self.param_training,
                                                      early_stoppers=early_stoppers, nb_split=1, shuffle_kfold=True,
                                                      percent_val_for_1_fold=0, silent=True)

            best_loss = estimator_history.get_best_value_for('loss_training')
            # : fetch the best value and assert if accuracy > threshold.
            assert best_loss < 0.01

    def test_errors_input(self):
        try:
            (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                      param_train=self.param_training,
                                                      early_stoppers=self.early_stoppers, nb_split=1,
                                                      shuffle_kfold=True,
                                                      percent_val_for_1_fold=-5, silent=True)
            PermissionError('Should have broke there, and the error caught! The error should have been: '
                            'AssertionError: percent_validation_for_1_fold should be in [0,100[ !')

        except AssertionError:
            pass
        try:
            (net, estimator_history) = nn_kfold_train(self.train_X, self.train_Y, self.Class_Parametrized_NN,
                                                      param_train=self.param_training,
                                                      early_stoppers=self.early_stoppers, nb_split=1,
                                                      shuffle_kfold=True,
                                                      percent_val_for_1_fold=150, silent=True)
            PermissionError('Should have broke there, and the error caught! The error should have been: '
                            'AssertionError: percent_validation_for_1_fold should be in [0,100[ !')

        except AssertionError:
            pass
