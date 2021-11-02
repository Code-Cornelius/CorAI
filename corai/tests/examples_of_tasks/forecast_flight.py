import numpy as np
import priv_lib_ml as corai
import seaborn as sns
import torch
import torch.nn as nn
from priv_lib_plot import APlot
from sklearn.preprocessing import MinMaxScaler

# set seed for pytorch.
corai.set_seeds(42)

# Import Data
flight_data = sns.load_dataset("flights")

##########################################  GLOBAL PARAMETERS
device = corai.pytorch_device_setting('gpu')
SILENT = False
early_stop_train = corai.Early_stopper_training(patience=400, silent=SILENT, delta=-int(1E-2))
early_stop_valid = corai.Early_stopper_validation(patience=400, silent=SILENT, delta=-int(1E-2))
early_stoppers = (early_stop_train, early_stop_valid)
metrics = ()
##########################################  DATA PARAMETERS:
time_series_len = lookback_window = 18
# time_series_len = lookback_window = 12
lookforward_window = 12
nb_test_prediction = lookback_window // lookforward_window + 1
nb_unknown_prediction = 36 // lookforward_window
##########################################  main
if __name__ == '__main__':
    # config of the architecture:
    input_size = 3
    num_layers = 4
    bidirectional = True
    hidden_size = 128
    output_size = 1
    dropout = 0.01
    epochs = 80000
    batch_size = 120
    hidden_FC = 128

    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": 0.00005, "weight_decay": 1E-6}
    optim_wrapper = corai.Optim_wrapper(optimiser, dict_optimiser)

    param_training = corai.NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                             criterion=criterion, optim_wrapper=optim_wrapper,
                                             metrics=metrics)

    Parent = corai.GRU
    rnn_class = nn.GRU

    seq_nn = [
        (model := corai.factory_parametrised_RNN(input_dim=input_size, output_dim=output_size,
                                                 num_layers=num_layers, bidirectional=bidirectional,
                                                 input_time_series_len=lookback_window,
                                                 output_time_series_len=lookforward_window,
                                                 nb_output_consider=lookforward_window,
                                                 hidden_size=hidden_size, dropout=dropout,
                                                 Parent=Parent, rnn_class=rnn_class)()),  # walrus operator
        corai.Reshape([-1, model.output_len]),
        nn.Linear(model.output_len, hidden_FC, bias=True),
        nn.CELU(),
        nn.Linear(hidden_FC, lookforward_window * output_size, bias=True),
        corai.Reshape([-1, lookforward_window, output_size]),
    ]

    parametrized_NN = corai.factory_parametrised_Free_NN(seq_nn)

    ########################################## DATA
    print(flight_data.head())
    flight_data['time_month'] = range(len(flight_data))
    flight_data = flight_data.drop(columns=["year", "month"])
    # add cyclicity
    flight_data = corai.add_column_cyclical_features(flight_data, 'time_month', 12)
    print(flight_data.head())

    if input_size == 1:
        data = flight_data['passengers'].values.astype(float).reshape(-1, 1)
    else:
        data = flight_data.values.astype(float)

    train_data = data[:-lookback_window].reshape(-1, input_size)
    testing_data = data[-lookback_window:].reshape(-1, input_size)

    minimax = MinMaxScaler(feature_range=(-1., 1.))
    train_data_normalized = torch.FloatTensor(minimax.fit_transform(train_data))
    testing_data_normalised = torch.FloatTensor(minimax.transform(testing_data))
    testing_data = torch.FloatTensor(testing_data)

    window = corai.Windowcreator(input_dim=input_size, output_dim=1, lookback_window=lookback_window,
                                 lag_last_pred_fut=lookforward_window,
                                 lookforward_window=lookforward_window, type_window="Moving")
    (data_training_X, data_training_Y) = window.create_input_sequences(train_data_normalized,
                                                                       train_data_normalized[:, 0].unsqueeze(1))

    indices_train = torch.arange(len(data_training_X) - lookback_window)
    indices_valid = torch.arange(len(data_training_X) - lookback_window, len(data_training_X))
    print("shape of training : ", data_training_Y.shape)


    ##########################################

    def inverse_transform(arr):
        return minimax.inverse_transform(np.array(arr).reshape(-1, input_size))


    ##########################################  TRAINING

    estimator_history = corai.Estim_history(metric_names=[], validation=True)
    net, _ = corai.train_kfold_a_fold_after_split(data_training_X, data_training_Y, indices_train, indices_valid,
                                                  parametrized_NN, param_training, estimator_history,
                                                  early_stoppers=early_stoppers)

    net.to(torch.device('cpu'))

    history_plot = corai.Relplot_history(estimator_history)
    history_plot.lineplot(log_axis_for_loss=True)


    class Adaptor_output(object):
        def __init__(self, incrm, start_num, period):
            self.incrm = incrm
            self.start_num = start_num
            self.period = period

        def __call__(self, arr):
            times = np.arange(self.incrm, self.incrm + lookforward_window)
            cos_val = np.sin(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
            sin_val = np.cos(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
            self.incrm += lookforward_window
            res = np.concatenate((arr.reshape(-1, 1), cos_val, sin_val), axis=1).reshape(1, -1, 3)
            return torch.tensor(res, dtype=torch.float32)


    ##########################################  prediction TESTING :
    # prediction by looking at the data we know about
    if input_size > 1:
        increase_data_for_pred = Adaptor_output(0, 0, 12)  # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    train_prediction = window.prediction_over_training_data(net, train_data_normalized, increase_data_for_pred,
                                                            device=device)

    if input_size > 1:
        increase_data_for_pred = Adaptor_output(train_data_normalized.shape[0],
                                                0, 12)  # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    ##########################################  prediction unknown set. Corresponds to predicting the black line.
    test_prediction = window.prediction_recurrent(net, train_data_normalized[-lookback_window:],
                                                  nb_test_prediction, increase_data_for_pred, device='cpu')

    ##########################################  prediction of TESTING unknown data by starting with black line.
    if input_size > 1:
        increase_data_for_pred = Adaptor_output(train_data_normalized.shape[0] + testing_data_normalised.shape[0],
                                                0, 12)  # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    unknwon_prediction = window.prediction_recurrent(net, testing_data_normalised[-lookback_window:, :],
                                                     nb_unknown_prediction, increase_data_for_pred, device='cpu')

    # x-axis data for plotting
    months_total = np.arange(0, len(data), 1)  # data for testing
    months_train = np.arange(0, len(train_data), 1)  # data for training
    months_train_prediction = np.arange(lookback_window, train_prediction.shape[1] + lookback_window, 1)
    length_training = len(data_training_Y) + lookback_window + lookforward_window - 1
    months_test = np.arange(length_training,
                            length_training + nb_test_prediction * lookforward_window, 1)
    months_forecast = np.arange(length_training + lookback_window,
                                length_training + lookback_window + lookforward_window * nb_unknown_prediction,
                                1)

    aplot = APlot()
    dict_ax = {'title': 'Data and Prediction during Training and over Non Observed Data.', 'xlabel': 'month',
               'ylabel': 'passenger'}
    dict_plot_param = {'label': 'Data for Testing', 'color': 'black', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_total, data[:, 0], dict_ax=dict_ax, dict_plot_param=dict_plot_param)
    dict_plot_param = {'label': 'Data Known at Training Time', 'color': 'gray', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_train, train_data[:, 0], dict_ax=dict_ax, dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction over Training', 'color': 'royalblue', 'linestyle': '--', 'linewidth': 2}
    aplot.uni_plot(0, months_train_prediction, inverse_transform(train_prediction)[:, 0],
                   dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction over Test Set', 'color': 'r', 'linestyle': '--', 'linewidth': 1.5}
    aplot.uni_plot(0, months_test, inverse_transform(test_prediction)[:, 0], dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction of Future Unknown Set', 'color': 'g', 'linestyle': '--', 'linewidth': 1.5}
    aplot.uni_plot(0, months_forecast, inverse_transform(unknwon_prediction)[:, 0], dict_plot_param=dict_plot_param)
    aplot.show_legend()
    APlot.show_plot()
