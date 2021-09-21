import numpy as np
import torch
from tqdm import tqdm


class Windowcreator(object):
    """
    Semantics:
        Helper class to create windows for prediction.
        The object has to be used when X and Y datasets have the same length:
            x1, x2, ... xn corresponds to a yn+1, ... ym.

    References:
        https://www.tensorflow.org/tutorials/structured_data/time_series#2_split

    todo a bit too much for forecasting.

    todo not sure how increasing window work...
    """

    def __init__(self, input_dim, output_dim,
                 lookback_window,
                 lookforward_window=0, lag_last_pred_fut=1,
                 type_window="Moving",
                 batch_first=True, silent=False):

        assert type_window == "Increasing" or type_window == "Moving", "Only two types supported."
        assert not (type_window == "Increasing" and lookback_window != 0), "Increasing so window ==0."
        assert not (type_window == "Moving" and lookback_window == 0), "Moving so window > 0."

        # Window parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback_window = lookback_window
        self.lookforward_window = lookforward_window
        self.type_window = type_window
        self.lag_last_pred_fut = lag_last_pred_fut

        self.batch_first = batch_first

        self.silent = silent

        # Parameters of the slices
        self.complete_window_data = self.lookback_window + self.lag_last_pred_fut

        self.input_slice = slice(0, self.lookback_window)
        self.input_indices = np.arange(self.complete_window_data)[self.input_slice]

        self.index_start_prediction = self.complete_window_data - self.lookforward_window
        self.slices_prediction = slice(self.index_start_prediction, None)  # None means to the end
        self.indices_prediction = np.arange(self.complete_window_data)[self.slices_prediction]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.complete_window_data}',
            f'X indices: {self.input_indices}',
            f'Y indices: {self.indices_prediction}'])

    def create_input_sequences(self, input_data, output_data):
        """
        Semantics:
            create the dataset for training.

        Args:
            input_data (pytorch tensor): should be a N*M matrix, column is a time series.
            output_data (pytorch tensor): should be a N'*M' matrix, column is a time series.

        Returns:
            Two tensors with the data split in this shape:
                [batch size, sequence, dim output]

        References :
            from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

        """
        L = len(input_data)  # shape[0]
        nb_of_data = L - self.complete_window_data + 1  # nb of data - the window, but there is always one data so +1.

        assert self.lookback_window < L, \
            f"lookback window is bigger than data. Window size : {self.lookback_window}, Data length : {L}."
        assert self.lookforward_window < L, \
            f"lookforward window is bigger than data. Window size : {self.lookback_window}, Data length : {L}."
        assert len(input_data) == len(output_data)

        if self.batch_first:  # specifies how to take the input
            data_X = torch.zeros(nb_of_data, self.lookback_window, self.input_dim)
            data_Y = torch.zeros(nb_of_data, self.lookforward_window, self.output_dim)

            for i in tqdm(range(nb_of_data), disable=self.silent):
                data_X[i, :, :] = input_data[i:i + self.lookback_window, :].view(1, self.lookback_window,
                                                                                 self.input_dim)
                slice_out = slice(i + self.lookback_window, i + self.lookback_window + self.lookforward_window)
                data_Y[i, :, :] = output_data[slice_out, :].view(1, self.lookforward_window, self.output_dim)
            return data_X, data_Y

        else:
            data_X = torch.zeros(self.lookback_window, nb_of_data, self.input_dim)
            data_Y = torch.zeros(nb_of_data, self.input_dim, self.output_dim)

            for i in tqdm(range(nb_of_data), disable=self.silent):
                data_X[:, i, :] = input_data[i:i + self.lookback_window, :].view(self.lookback_window, self.input_dim)
                data_Y[:, i, :] = output_data[i + self.lookback_window: i + self.lookback_window +
                                                                        self.lookforward_window, :]
            return data_X, data_Y

    def prediction_over_training_data(self, net, data_start, increase_data_for_pred, device):
        """
        Semantics:
            predict the output by taking the input data and iterating over it by the window.

        Args:
            net:
            data_start:
            increase_data_for_pred:
            device (): where the net lies.


        Returns:

        """
        # a container has the lookback window (at least) of data.
        # Then iteratively, it predicts the future.
        # data_start should be not prepared dataset
        # format L * dim_input

        # we predict by each window of prediction, which is what seems to have the more sense.
        assert self.lookback_window <= len(data_start), "For prediction, needs at least a window of data for prediction"

        nb_of_cycle_pred = (len(data_start) - self.lookback_window) // self.lookforward_window
        prediction = torch.zeros(1, self.lookforward_window * nb_of_cycle_pred, self.input_dim)
        for i in range(nb_of_cycle_pred):
            indices_input = slice(i * self.lookforward_window, i * self.lookforward_window + self.lookback_window)
            #  : we start at the lookforward_window * i and need lookback_window elements.
            indices_pred = slice(i * self.lookforward_window, (i + 1) * self.lookforward_window)
            new_values = net.nn_predict(data_start[indices_input, :].view(1, -1, self.input_dim))
            # the view for the batch size.

            new_values = self.adding_input_to_output(increase_data_for_pred, new_values, device)

            prediction[0, indices_pred, :] = new_values
        return prediction

    def adding_input_to_output(self, increase_data_for_pred, new_values, device):
        if increase_data_for_pred is not None:
            new_values = increase_data_for_pred(new_values.cpu().numpy()).to(device)
            # cpu to make sure, numpy to avoid implicit conversion.
        return new_values

    def prediction_recurrent(self, net, data_start, nb_of_cycle_pred, increase_data_for_pred=None, device='cpu'):
        """

        Args:
            net:
            data_start:
            nb_of_cycle_pred:
            increase_data_for_pred:
            device:  where data_start lies.

        Returns:

        """
        # a container has the lookback window (at least) of data.
        # Then iteratively, it predicts the future.
        # data_start should be not prepared dataset
        # format L * dim_input
        # increase data in the case the output is not exactly the input for next prediction!
        assert self.lookback_window == len(data_start), "For prediction, needs at least a window of data for prediction"

        input_prediction = data_start.clone()
        prediction = torch.zeros(1, self.lookforward_window * nb_of_cycle_pred, self.output_dim)

        for i in range(nb_of_cycle_pred):
            indices_in = slice(i * self.lookforward_window, i * self.lookforward_window + self.lookback_window)
            #  : we start at the lookforward_window * i and need lookback_window elements.
            indices_pred = slice(i * self.lookforward_window, (i + 1) * self.lookforward_window)
            new_values = net.nn_predict(input_prediction[indices_in, :].view(1, -1, self.input_dim))
            prediction[0, indices_pred, :] = new_values

            new_values = self.adding_input_to_output(increase_data_for_pred, new_values, device)

            input_prediction = torch.cat((input_prediction, new_values.view(-1, self.input_dim)))
            # the view for the batch size.
        return input_prediction[self.lookback_window:].view(1, -1, self.input_dim)


"""
example of increase_data_for_pred:

class Adaptor_output(object):
    def __init__(self, incrm, start_num, period):
        self.incrm = incrm
        self.start_num = start_num
        self.period = period

    def __call__(self, arr):
        times = np.arange(self.incrm, self.incrm + output_time_series_len)
        cos_val = np.sin(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
        sin_val = np.cos(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
        self.incrm += input_time_series_len
        res = np.concatenate((arr.reshape(-1, 2), cos_val, sin_val), axis=1).reshape(1, -1, 4)
        return torch.tensor(res, dtype=torch.float32)
"""
