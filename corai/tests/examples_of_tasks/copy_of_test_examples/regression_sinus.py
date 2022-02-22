import torch
from torch import nn

import corai
from corai.tests.sinus_dataset_generator import data_sinus
from corai_plot import APlot

# set seed for pytorch.
corai.set_seeds(42)

train_X, train_Y, testing_X, testing_Y, plot_xx, plot_yy, plot_yy_noisy, xx, yy = data_sinus()

device = corai.pytorch_device_setting('cpu')
SILENT = False

if __name__ == '__main__':
    # config of the architecture:
    input_size = 1
    hidden_sizes = [4, 8, 4]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.relu]
    dropout = 0.
    epochs = 7500
    batch_size = 200
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": 0.01, "weight_decay": 0.0000001}
    optim_wrapper = corai.Optim_wrapper(optimiser, dict_optimiser)


    def L4loss(net, xx, yy):
        return torch.norm(net.nn_predict(xx) - yy, 4)


    L4metric = corai.Metric('L4', L4loss)
    metrics = (L4metric,)
    param_training = corai.NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                             criterion=criterion, optim_wrapper=optim_wrapper,
                                             metrics=metrics)
    Class_Parametrized_NN = corai.factory_parametrised_FC_NN(param_input_size=input_size,
                                                             param_list_hidden_sizes=hidden_sizes,
                                                             param_output_size=output_size, param_list_biases=biases,
                                                             param_activation_functions=activation_functions,
                                                             param_dropout=dropout,
                                                             param_predict_fct=None)

    early_stop_train = corai.Early_stopper_training(patience=20, silent=SILENT, delta=-1E-6)
    early_stop_valid = corai.Early_stopper_validation(patience=20, silent=SILENT, delta=-1E-6)
    early_stoppers = (early_stop_train, early_stop_valid)
    (net, estimator_history) = corai.nn_kfold_train(train_X, train_Y, Class_Parametrized_NN, param_train=param_training,
                                                    early_stoppers=early_stoppers, nb_split=1, shuffle_kfold=True,
                                                    percent_val_for_1_fold=10, silent=False)

    history_plot = corai.Relplot_history(estimator_history)
    history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot='L4', log_axis_for_loss=True,
                                            log_axis_for_second_axis=True)
    history_plot.lineplot(log_axis_for_loss=True)

    corai.nn_plot_prediction_vs_true(net=net, plot_xx=plot_xx,
                                     plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy,
                                     device=device)
    corai.nn_errors_compute_mean(net=net, train_X=train_X, train_Y=train_Y, testing_X=testing_X, testing_Y=testing_Y,
                                 device=device)
    print(estimator_history)
    APlot.show_plot()
