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

    Todos:
        todo not sure how increasing window work...
    """

    def __init__(self, input_dim, output_dim,
                 lookback_window,
                 lookforward_window=0, lag_last_pred_fut=1,
                 type_window="Moving",
                 batch_first=True, silent=False):
        """Look inside the class to see what the parameters correspond to."""

        assert type_window == "Increasing" or type_window == "Moving", "Only two types supported."
        assert not (type_window == "Increasing" and lookback_window != 0), "Increasing so window ==0."
        assert not (type_window == "Moving" and lookback_window == 0), "Moving so window > 0."

        assert lookforward_window <= lag_last_pred_fut, "lag_last_pred_fut is at least as long as lookforward_window."
        # Window parameters.
        self.input_dim = input_dim  # dimension of input
        self.output_dim = output_dim  # dimension of output
        self.lookback_window = lookback_window  # how many steps in the back input should have.
        self.lookforward_window = lookforward_window  # how many steps in the future should be predicted.
        self.type_window = type_window  # type of windows
        self.lag_last_pred_fut = lag_last_pred_fut  # difference in time between last known data and last prediction.
        # in other words,  lag_last_pred_fut = t+h - t where t+h is predicted, t is known.
        # lag_last_pred_fut might be different from lookforward window if we are not predicting the +1 step, but the +2 step.
        # in this example: u[0],u[1],u[2]. We predict u[4].
        # Then,  lag_last_pred_fut = 2; lookforward_window = 1.

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
            create the dataset for training. Give time-series as u[t], v[t], without any time difference.
            They should be of matching size, and the function will deduce which values correspond to which, by the windows given at initialisation.

        Args:
            input_data  (pytorch tensor (batch N, length L, D_in nb dimensions) if batch first): all time-series from batch must have the same length,
                because otherwise they would not fit inside a tensor.
            output_data (pytorch tensor (batch N, length L, D_out nb dimensions)):

        Returns:
            Two tensors with the data split in this shape:
                [batch size, sequence, dim output]

        References :
            from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

        """
        nb_batch = input_data.shape[0]  # N as in documentation.
        L = input_data.shape[1]  # L as in documentation.

        nb_data = L - self.complete_window_data + 1  # nb of data - the window, but there is always one data so +1.
        # nb_data represents the amount of different input to the learning algo.

        assert self.lookback_window < L, \
            f"lookback window is not smaller than data. Window size : {self.lookback_window}, Data length : {L}."
        assert self.lookforward_window < L, \
            f"lookforward window is not smaller than data. Window size : {self.lookback_window}, Data length : {L}."

        assert input_data.shape[0] == output_data.shape[0], \
            f"Batch size not matching {input_data.shape[0]}, {output_data.shape[0]}."
        assert input_data.shape[1] == output_data.shape[1], \
            f"Time-series length not matching {input_data.shape[1]}, {output_data.shape[1]}."

        # in the following, we have data as nb_batch and as nb_data.
        # nb_data corresponds to how many samples of observation we create out of
        # one time series from the batches we gave (first component).
        # but we do the same for all time-series. Hence the nb_batch.
        # However we do not care in the end what data comes from what batch and we flatten the dimensions together.
        if self.batch_first:  # specifies how to take the input
            data_X = torch.zeros(nb_batch, nb_data, self.lookback_window, self.input_dim)
            data_Y = torch.zeros(nb_batch, nb_data, self.lookforward_window, self.output_dim)

            for i in tqdm(range(nb_data), disable=self.silent):
                data_X[:, i, :, :] = input_data[:,
                                     i:i + self.lookback_window, :]  # add dimension of nb_data
                slice_out = slice(i + self.lookback_window, i + self.lookback_window + self.lookforward_window)
                data_Y[:, i, :, :] = output_data[:, slice_out, :]  # add dimension of nb_data

            data_X = torch.flatten(data_X, start_dim=0, end_dim=1)
            data_Y = torch.flatten(data_Y, start_dim=0, end_dim=1)
            return data_X, data_Y

        else:
            data_X = torch.zeros(self.lookback_window, nb_batch, nb_data, self.input_dim)
            data_Y = torch.zeros(nb_batch, nb_data, self.lookforward_window, self.output_dim)

            for i in tqdm(range(nb_data), disable=self.silent):
                data_X[:, :, i, :] = input_data[i:i + self.lookback_window]
                slice_out =  slice(i + self.lookback_window, i + self.lookback_window + self.lookforward_window)
                data_Y[:, i, :, :] = output_data[:, slice_out, :]

            data_X = torch.flatten(data_X, start_dim=1, end_dim=2)
            data_Y = torch.flatten(data_Y, start_dim=0, end_dim=1)
            return data_X, data_Y

    def prediction_over_training_data(self, net, data, increase_data_for_pred, device):
        """
        Semantics:
            predict the output by taking the input data and iterating over data by the window.
            Used to make prediction iteratively over known input.
            In order to add the data that has not been forecasted (for example you predict 2D -> 1D,
            the output is missing 1D for the future predictions), we use an adaptor. It is the parameter increase_data_for_pred.

        Args:
            net (Savable_net): model with the method nn_predict.
            data: tensor with shape (N batch, L length time series, Dim Input)
            increase_data_for_pred (callable):  Using a class for this allows to store some parameters.
            device (pytorch device): where the net lies with data.

        Returns:

        """
        # a container has the lookback window (at least) of data.
        # data_start should be not prepared dataset
        # format L * dim_input

        # we predict batches of window of prediction, and then going forward by this batch size for next estim.
        assert self.lookback_window <= data.shape[1], "For prediction, needs at least a window of data for prediction"

        nb_of_cycle_pred = (data.shape[1] - self.lookback_window) // self.lookforward_window
        prediction = torch.zeros(1, self.lookforward_window * nb_of_cycle_pred, self.input_dim)
        for i in range(nb_of_cycle_pred):
            indices_input = slice(i * self.lookforward_window,
                                  i * self.lookforward_window + self.lookback_window)
            #  : we start at the lookforward_window * i and we need lookback_window elements.
            indices_pred = slice(i * self.lookforward_window,
                                 (i + 1) * self.lookforward_window)

            # predicting the next values, size (1,-1,self.input_dim) reflects (batch_size, length pred, self.input_dim)
            new_values = net.nn_predict(data[:, indices_input, :])
            # add values to the prediction:
            new_values = self._adding_input_to_output(increase_data_for_pred, new_values, device)

            prediction[:, indices_pred, :] = new_values
        return prediction

    def prediction_recurrent(self, net, data_start, nb_of_cycle_pred, increase_data_for_pred=None, device='cpu'):
        """
        Semantics:
            Prediction by iteratively using the previous prediction as inputs for the following ones.
            In order to add the data that has not been forecasted (for example you predict 2D -> 1D,
            the output is missing 1D for the future predictions), we use an adaptor. It is the parameter increase_data_for_pred.
        Args:
            net (Savable_net): model with the method nn_predict.
            data_start: starting data, to initialise the recurrent process. tensor with shape (N batch, L length time series, Dim Input)
            nb_of_cycle_pred:
            increase_data_for_pred (callable):  Using a class for this allows to store some parameters.
            device (pytorch device): where the net lies with data_start.

        Returns:

        """
        # a container has the lookback window (at least) of data.
        # Then iteratively, it predicts the future.
        # data_start should be not prepared dataset
        # format L * dim_input
        # increase data in the case the output is not exactly the input for next prediction!
        assert self.lookback_window == data_start.shape[
            1], "For prediction, needs a window of data for prediction. Given {}.".format(data_start.shape[1])

        input_prediction = data_start.clone()
        prediction = torch.zeros(1, self.lookforward_window * nb_of_cycle_pred, self.output_dim)

        for i in range(nb_of_cycle_pred):
            indices_in = slice(i * self.lookforward_window,
                               i * self.lookforward_window + self.lookback_window)
            #  : we start at the lookforward_window * i and need lookback_window elements.
            indices_pred = slice(i * self.lookforward_window,
                                 (i + 1) * self.lookforward_window)

            # predicting the next values, size (1,-1,self.input_dim) reflects (batch_size, length pred, self.input_dim)
            new_values = net.nn_predict(input_prediction[:, indices_in, :])
            prediction[:, indices_pred, :] = new_values

            new_values = self._adding_input_to_output(increase_data_for_pred, new_values, device)
            input_prediction = torch.cat((input_prediction, new_values), dim=1)
        return input_prediction[:, self.lookback_window:]  # remove the starting time series

    def _adding_input_to_output(self, increase_data_for_pred, new_values, device):
        if increase_data_for_pred is not None:
            new_values = increase_data_for_pred(new_values.cpu().numpy()).to(device)
            # cpu to make sure, numpy to avoid implicit conversion.
        return new_values
