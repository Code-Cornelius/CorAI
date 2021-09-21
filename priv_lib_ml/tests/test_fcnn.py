from unittest import TestCase

import torch

from nn_classes.architecture.fully_connected import factory_parametrised_FC_NN

class Test_fcnn(TestCase):

    def test_nb_parameters(self):
        nn = factory_parametrised_FC_NN(param_input_size=784,
                                        param_list_hidden_sizes=[16, 16],
                                        param_output_size=10,
                                        param_list_biases=[True, True, True],
                                        param_activation_functions=[torch.tanh, torch.tanh])()


        assert nn.nb_of_params == 12960