# function applying DNNOPT

import numpy as np
# from a surrogate, find the minimiser to some loss function.
import pandas as pd
from torch import optim
from tqdm import tqdm


def DNNopt(optimal_values, loss_function, list_lr, nb_new_pts_to_add):
    """ first_guesses is a tensor. """

    optimals_values = [optimal_values.clone().requires_grad_(True) for _ in list_lr]
    list_optimizer = [optim.LBFGS([val],  # takes iterable of tensors
                                  lr=lr,
                                  max_iter=50000,
                                  max_eval=50000,
                                  history_size=200,
                                  tolerance_change=1.0 * np.finfo(float).eps)
                      for val, lr in zip(optimals_values, list_lr)]
    list_final_losses = []
    corresponding_optimals = []
    zap = zip(optimals_values, list_optimizer)
    for _ in tqdm(range(len(optimals_values))):
        val, optimizer = next(zap)

        def closure():
            optimizer.zero_grad()
            loss = loss_function(val)
            # print("loss: \n", loss)
            # print("optimal value: \n", val[0, :])
            loss.backward()
            return loss

        optimizer.step(closure=closure)
        loss = loss_function(val)
        loss = loss.item()
        list_final_losses.append(loss)
        corresponding_optimals.append(val.detach().numpy())

    dict_result = pd.DataFrame({"Loss": list_final_losses, "Optimal": corresponding_optimals})
    dict_result = dict_result.sort_values(by="Loss", ascending=True)
    # print("Results of optimisation: \n", dict_result)
    optimised = dict_result.iloc[0, 1]
    try:
        optimised = filter_values(optimised, nb_new_pts_to_add)
    except ValueError:
        try:
            print("First loss did not give any good result, trying the second one")
            optimised = dict_result.iloc[1, 1]  # second solution
            optimised = filter_values(optimised, nb_new_pts_to_add)
        except ValueError:
            try:
                print("First loss did not give any good result, trying the third one")
                optimised = dict_result.iloc[2, 1]  # third solution
                optimised = filter_values(optimised, nb_new_pts_to_add)
            except ValueError:  # We write message that failed
                with open("ERROR_MESSAGE.txt", "w") as file:
                    file.write("No good values found for next iteration.")
                raise ValueError
    # print("Final Optimisers:\n", optimised)
    return optimised


# Use the main

# chose the parameters and train all models.
# Once all models done, you pick the one with lowest validation loss.
# This one is the best, that is used to create new data points.

# create loss function with this new best model
# call ISMO
# ISMO has a file where he stores:
# current iteration, nb of points created between trainings
# it takes first_guesses, randomly generated outside, and the loss.
# change the two files (iterate the point i, and add new data points).

# be careful about the scaling !!!


# DNNopt(first_guesses, loss_function)

def read_list_of_ints_from_path(path):
    ans = []
    with open(path, "r") as file:
        for line in file:
            ans.append(float(line.strip()))
    return ans


def read_ismo_config(path):
    (iteration_nb, nb_new_pts_to_add, nb_pts_to_minimise,
     final_hand_back_points, final_computed_point, target_opti) = read_list_of_ints_from_path(path)
    (iteration_nb, nb_of_new_points_to_create,
     nb_of_new_points_to_compute, final_written_down_point,
     final_computed_point, target_opti) = (
        int(iteration_nb), int(nb_new_pts_to_add),
        int(nb_pts_to_minimise), int(final_hand_back_points),
        int(final_computed_point), target_opti)
    return (iteration_nb, nb_of_new_points_to_create, nb_of_new_points_to_compute,
            final_written_down_point, final_computed_point, target_opti)


def name_config_type(iteration_nb):
    return f"train_{iteration_nb}"


def filter_values(numpy_arr_two_col, nb_new_pts_to_add):
    # based on data
    extrem_value1 = -1.64  # I deducted 0.2 to all such that values are not so extreme.
    extrem_value2 = 1.21  # 0.5 to stay in range of values i know about
    extrem_value3 = -1.54
    extrem_value4 = 0
    cdt = ((extrem_value1 < numpy_arr_two_col[:, 0]) &
           (numpy_arr_two_col[:, 0] < extrem_value2) &
           (extrem_value3 < numpy_arr_two_col[:, 1]) &
           (numpy_arr_two_col[:, 1] < extrem_value4))
    numpy_arr_two_col = numpy_arr_two_col[cdt, :]
    if len(numpy_arr_two_col) < nb_new_pts_to_add:
        raise ValueError(f"Less points available ({len(numpy_arr_two_col)}) than the required number.")
    else:
        return numpy_arr_two_col[:nb_new_pts_to_add, :]


def ISMO(file_with_parameters_for_ismo, file_with_data_for_training, loss_fct):
    # take the best model, and use it for DNNopt.
    # first, we do it alltogether, nothing fancy. Just start with some initial points (nb given in file) and find the new points, that you add to the data.
    # write file.
    pass
