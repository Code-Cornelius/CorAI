from enum import Enum, auto

import torch
from torch import Tensor
from tqdm import tqdm


class WindowCreator(object):
    """
    Semantics:
        Helper class to create windows for prediction.
        Set the parameters of your dataset during initialisation and create the data with the methods.

    References:
        https://www.tensorflow.org/tutorials/structured_data/time_series#2_split

    """

    class AllowedType(Enum):
        MOVING = auto()
        INCREASING = auto()

        @staticmethod
        def condition_moving(lookback_window):
            return lookback_window == 0

        @staticmethod
        def condition_increasing(lookback_window):
            return lookback_window != 0

    def __init__(self, input_dim: int,
                 output_dim: int,
                 lookback_window: int,
                 lookforward_window: int = 0,
                 end_pred_window: int = 1,
                 window_type: AllowedType = AllowedType.MOVING,
                 batch_first: bool = True, silent: bool = False):
        """
        
        Args:
            input_dim: dimension of input
            output_dim: dimension of output
            lookback_window:  how many steps in the back the input should have. For example, if we try to predict v3, a window of 2 means we will use v1 and v2.
            lookforward_window: how far in the future do we predict. If we have knowledge until v3, a window of 3 means we predict v6.
            end_pred_window: difference of steps between the last known data's time and the last prediction's time.
                In other words,  end_pred_window := t+h - t where `t+h` represents the prediction's time using data until time `t`.
            window_type: type of windows.
            batch_first: 
            silent: 
        
        Examples:
            Assuming we have u[0],u[1],u[2]. We predict u[4]. Then, 
                end_pred_window = 2
                lookforward_window = 1
        """

        assert window_type in WindowCreator.AllowedType, \
            "type_window should be of type `AllowedType`."

        assert not (window_type == WindowCreator.AllowedType.INCREASING
                    and WindowCreator.AllowedType.condition_increasing(lookback_window)), \
            "Increasing so window == 0."

        assert not (window_type == WindowCreator.AllowedType.MOVING
                    and WindowCreator.AllowedType.condition_moving(lookback_window)), \
            "Moving so window > 0."

        assert (lookforward_window <= end_pred_window), \
            "end_pred_window is at least as long as lookforward_window. " \
            "It correspond to the maximal lag between the last seen data  and the maximum prediction required." \
            "There might be a mismatch if one requires a prediction at time +1."

        # Window parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback_window = lookback_window
        self.lookforward_window = lookforward_window
        self.type_window = window_type
        self.end_pred_window = end_pred_window
        self.batch_first = batch_first
        self.silent = silent

        # Parameters of the slices
        self.complete_window_data = self.lookback_window + self.end_pred_window

    def create_input_sequences(self, input_data: torch.tensor, output_data: torch.tensor = None):
        """
        Semantics:
            create the dataset for training. Give time-series as u[t], v[t], without any time difference. If there is, add padding to reflect it.
            They should be of matching size, and the function will deduce which values correspond to which, by the windows given at initialisation.

        Args:
            input_data  (pytorch tensor (batch N, length L, D_in nb dimensions) if batch first): all time-series from batch must have the same length,
                because otherwise they would not fit inside a tensor.
            output_data (pytorch tensor (batch N, length L, D_out nb dimensions)): can be none, then no output_data returned.

        Returns:
            Two tensors with the data split in this shape:
                [batch size, sequence, dimension]
            only data_x if no output_data passed.

        References :
            from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

        """
        # we return data with same type as input
        dtype = input_data.dtype
        if output_data is not None:
            assert dtype == output_data.dtype, "Data Input/Output should have the same dtype."

        nb_batch = input_data.shape[0]  # N as in documentation.
        L = input_data.shape[1]  # L as in documentation.

        # We will get length time series minus the window size plus one number of points.
        nb_data = L - self.complete_window_data + 1

        self._assert_cdt_create_sequences(L, input_data, output_data)

        # Assuming our data is two time series stacked (u1,u2,u3;v1,v2,v3), with shape (2,3,D_in), then for all time series,
        # we construct the different time series that will be used for training.
        # For example, using a lookback_window of 1, nb_data = 2 and:
        #   (u1,u2,u3;v1,v2,v3) -> (u1,u2;v1,v2), (u2,u3;v2,v3).
        # We flatten the resulting tensor along the batching dimension.
        data_X = torch.zeros(nb_batch, nb_data, self.lookback_window, self.input_dim, dtype=dtype)
        if output_data is not None:
            data_Y: Tensor = torch.zeros(nb_batch, nb_data, self.lookforward_window, self.output_dim, dtype=dtype)

        for i in tqdm(range(nb_data), disable=self.silent):
            data_X[:, i, :, :] = input_data[:, i:i + self.lookback_window, :]

            if output_data is not None:
                # For v1, v2, v3, v4, v5, v6. lookback = 2, lookforward = 2, end_pred_window = 3, then:
                # v1, v2, -> v4, v5,
                # v2, v3, -> v5, v6.
                slice_out = slice(i + self.lookback_window + self.end_pred_window - self.lookforward_window,
                                  i + self.lookback_window + self.end_pred_window)
                data_Y[:, i, :, :] = output_data[:, slice_out, :]  # add dimension of nb_data

        data_X = torch.flatten(data_X, start_dim=0, end_dim=1)

        if not self.batch_first:
            data_X = data_X.transpose(0, 1)  # Resulting tensor has dimension (sequence, batch, dim input).
        if output_data is not None:
            data_Y = torch.flatten(data_Y, start_dim=0, end_dim=1)
            return data_X, data_Y

        return data_X, None

    def _assert_cdt_create_sequences(self, sequence_len, input_data, output_data=None):
        assert self.lookback_window < sequence_len, \
            f"lookback window is not smaller than data. Window size : {self.lookback_window}, Data length : {sequence_len}."
        assert self.lookforward_window < sequence_len, \
            f"lookforward window is not smaller than data. Window size : {self.lookback_window}, Data length : {sequence_len}."
        assert input_data.shape[2] == self.input_dim, \
            f"Time-series input dimension not corresponding to the window's: {input_data.shape[2]}, {self.input_dim}."
        if output_data is not None:
            assert input_data.shape[0] == output_data.shape[0], \
                f"Batch size not matching: {input_data.shape[0]}, {output_data.shape[0]}."
            assert input_data.shape[1] == output_data.shape[1], \
                f"Time-series length not matching: {input_data.shape[1]}, {output_data.shape[1]}."
            assert output_data.shape[2] == self.output_dim, \
                f"Time-series output dimension not corresponding to the window's: {output_data.shape[2]}, {self.output_dim}."

    # WIP IS THIS CORRECT
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

    # WIP IS THIS CORRECT
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
        assert self.lookback_window == data_start.shape[1], \
            "For prediction, needs a window of data for prediction. Given {}.".format(data_start.shape[1])

        input_prediction = data_start.clone()
        prediction = torch.zeros(1, self.lookforward_window * nb_of_cycle_pred, self.output_dim)

        for i in tqdm(range(nb_of_cycle_pred)):
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

    # WIP IS THIS CORRECT
    def _adding_input_to_output(self, increase_data_for_pred, new_values, device):
        if increase_data_for_pred is not None:
            new_values = increase_data_for_pred(new_values.cpu().numpy()).to(device)
            # cpu to make sure, numpy to avoid implicit conversion.
        return new_values
