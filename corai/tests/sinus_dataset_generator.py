import numpy as np
import torch

import corai

# set seed for pytorch.

corai.set_seeds(42)


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


def data_sinus():
    ############################## GLOBAL PARAMETERS
    n_samples = 2000  # Number of training samples
    sigma = 0.01  # Noise level
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

    return train_X, train_Y, testing_X, testing_Y, plot_xx, plot_yy, plot_yy_noisy, xx, yy
