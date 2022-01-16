import os

import corai
from corai import create_model_by_index
from corai_util.tools import function_dict
from corai_util.tools import function_writer

from datasets.dataset import *
from hp_opti_fct import generate_estims_history
from src.rc_class.full_model.ridge_models.model_esn_ridge import Model_esn_ridge
from src.rc_class.matrix_gen import *
from src.rc_class.plot_util import plot_prediction
from src.rc_class.util import *
from src.rc_class.windowcreatorrc import WindowCreatorRC

# section ######################################################################
#  #############################################################################
#  CHOICE global parameters
NEW_DATASET = False
ROOTPATH = os.path.dirname(os.path.abspath(__file__))  # get current location
linker_estims = function_writer.factory_fct_linked_path(ROOTPATH, "hp_optimisation_saving/estims")
linker_models = function_writer.factory_fct_linked_path(ROOTPATH, "hp_optimisation_saving/models")
PATH_JSON_PARAMS = os.path.join(ROOTPATH, "hp_optimisation_saving", "param_hyper_param_tuning.json")

device = corai.pytorch_device_setting('cpu')
corai.set_seeds(42)  # reset later
SILENT = False
dtype = torch.float32
washout = 50
lookforward_window = 1
nb_sample_point_ts = 800
model_choice = 3
hidden_dim = 64
dim, nb_dim = [1], 1
nb_batch = 1

# section ######################################################################
#  #############################################################################
#  CREATE DATASET
train_length = int(nb_sample_point_ts * 0.8)  # 80%
test_length = nb_sample_point_ts - train_length
plotting_points_starting_point = int(train_length * 0.5)  # 50% of the training set.
pre_scaled_inputs, pre_scaled_outputs = simple_equation_of_input((nb_batch, nb_sample_point_ts, nb_dim), dim)

#  SPLIT DATA
(inputs_train, outputs_train, inputs_test,
 outputs_test) = split_train_validation(train_length, pre_scaled_inputs, pre_scaled_outputs)
input_dim = inputs_train.shape[2]
output_dim = outputs_train.shape[2]

# Rescaling
minimax_in = MinMaxScaler(feature_range=(-1., 1.))
minimax_out = MinMaxScaler(feature_range=(-1., 1.))

inputs_train_scaled = flatten_batches_timeseries4scaling_back_normal(minimax_in, inputs_train, True)
outputs_train_scaled = flatten_batches_timeseries4scaling_back_normal(minimax_out, outputs_train, True)
inputs_test_scaled = flatten_batches_timeseries4scaling_back_normal(minimax_in, inputs_test, False)
outputs_test_scaled = flatten_batches_timeseries4scaling_back_normal(minimax_out, outputs_test, False)

WindowCreatorRC.assert_length(washout, lookforward_window, train_length, test_length)  # assert length are correct
# do not change the parameters lookback_windows and lookforward_window, as the code is not adapted for it.
wc = WindowCreatorRC(input_dim=input_dim, output_dim=output_dim, lookback_window=1,
                     lookforward_window=lookforward_window, lag_last_pred_fut=lookforward_window,
                     type_window="Moving", batch_first=True, washout=washout)

inputs_train_scaled, outputs_train_scaled = wc.create_input_sequences(inputs_train_scaled, outputs_train_scaled)
inputs_test_scaled, outputs_test_scaled = wc.create_input_sequences(inputs_test_scaled, outputs_test_scaled)


# section ######################################################################
#  #############################################################################
#  helper classes:
def L2loss(net, xx, yy):
    return torch.norm(net.nn_predict(xx) - yy[net.washout:], 2)


def L4loss(net, xx, yy):
    return torch.norm(net.nn_predict(xx) - yy[net.washout:], 4)


L2metric = corai.Metric('L2', L2loss)
L4metric = corai.Metric('L4', L4loss)
metrics = (L2metric, L4metric,)

mapping_names2functions = {
    # type of matrix:
    'uniform': matrix_gauss_gen.Matrix_gauss_gen,
    'gaussian': matrix_uni_gen.Matrix_uni_gen,
    # type of models
    'RNN Custom Fct': Model_esn_ridge,
    # non lin fct
    'tanh': torch.tanh
}

# section ######################################################################
#  #############################################################################
#  parameters for grid search

int_logspace = lambda beg, end, step: [int(x) for x in
                                       np.logspace(beg, end,
                                                   num=end - beg + 1,
                                                   base=2.0)[::step]]  # cast to int
rounded_linspace = lambda beg, end, nb_step: list(np.round(np.linspace(beg, end, nb_step), 2))
params_options_general = {
    "reservoir size": int_logspace(2, 7, 2),
    "dist W": ['uniform', 'gaussian'],
    # "dist Win": ['uniform', 'gaussian'],
    # "dist Wbias": ['uniform', 'gaussian'],
    # "dist h0": ['uniform', 'gaussian'],

    "scale W": list(np.arange(0.4, 1.6, 0.2)),
    # "scale Win": np.arange(0.4, 1.6, 0.1),
    # "scale Wbias": np.arange(0.4, 1.6, 0.1),
    # "scale h0": np.arange(0.4, 1.6, 0.1),

    "sparsity W": rounded_linspace(0.0, 0.9, 5),
    # "sparsity Win": np.linspace(0.0, 0.9, 3),
    # "sparsity Wbias": np.linspace(0.0, 0.9, 3),
    # "sparsity h0": np.linspace(0.0, 0.9, 3),
    "washout": [50],
    'seed': [42],
}

params_esn = params_options_general.copy()
params_esn['type RC'] = ['RNN Custom Fct']
params_esn['non_lin_fct'] = ['tanh']
params_esn["leak rate"] = [0.1, 0.4, 0.6, 0.9]

# convert parameters to the product of the parameters
hyper_params_leaky = function_dict.parameter_product(params_esn)
hyper_params_total = hyper_params_leaky  # merging

if NEW_DATASET:  # save the parameters
    function_writer.list_of_dicts_to_json(hyper_params_total, file_name=PATH_JSON_PARAMS, compress=False)
    print(f"File {PATH_JSON_PARAMS} has been updated.")
    print(f"    Number of configurations: {len(hyper_params_total)}.")


def config_architecture(params, input_dim, output_dim):
    # TODO 15/11/2021 nie_k:  efficient way of doing this

    print(params)
    params = params.copy()
    function_dict.replace_function_names_to_functions(params, mapping_names2functions, silent=True)

    reservoir_size = params["reservoir size"]
    washout = params["washout"]

    scale_W = params["scale W"]
    # scale_Win = params["scale Win"]
    # scale_Wbias = params["scale Wbias"]
    # scale_h0 = params["scale h0"]

    sparsity_W = params["sparsity W"]
    # sparsity_Win = params["sparsity Win"]
    # sparsity_Wbias = params["sparsity Wbias"]
    # sparsity_h0 = params["sparsity h0"]

    distribution_W = params["dist W"]
    # distribution_Win = params["dist Win"]
    # distribution_Wbias = params["dist Wbias"]
    # distribution_h0 = params["dist h0"]

    W = distribution_W(scale=scale_W, sparse=sparsity_W)

    if params['type RC'] == Model_esn_ridge:
        leaky_rate = params["leak rate"]

        non_lin_fct = params['non_lin_fct']

        model = params['type RC'](input_dim=input_dim, hidden_dim=reservoir_size, output_dim=output_dim,
                                  w_generator=W, win_generator=W, wbias_generator=W,
                                  h0_Generator=matrix_gauss_gen.Matrix_gauss_gen,
                                  h0_params={'scale': scale_W, 'sparse': sparsity_W},
                                  learning_algo='inv', ridge_param=1E-3, washout=washout, dtype=dtype,
                                  nonlin_fct=non_lin_fct, leak_rate=leaky_rate)
    else:
        raise ValueError('type RC')
    return model


# section ######################################################################
#  #############################################################################
#  COMPARISON MODELS

if NEW_DATASET:
    generate_estims_history(hyper_params_total, inputs_train_scaled, outputs_train_scaled,
                            inputs_test_scaled, outputs_test_scaled, config_architecture,
                            linker_estims, linker_models, metrics, SILENT)

estim_hyper_param = corai.Estim_hyper_param.from_folder(linker_estims(['']),
                                                        metric_names=["L2_training", "L4_training",
                                                                      "L2_validation", "L4_validation"],
                                                        flg_time=True, compressed=False)
histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
histplot_hyperparam.hist(column_name_draw='train_time', hue='reservoir size',
                         palette='RdYlBu', bins=20, separators_plot=None,
                         binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)
histplot_hyperparam.hist(column_name_draw='L2_training', hue='reservoir size',
                         palette='RdYlBu', bins=20, separators_plot=None,
                         binrange=None, stat='count', multiple="dodge", kde=False, path_save_plot=None)

scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
scatplot_hyperparam.scatter(column_name_draw='L2_training',
                            second_column_to_draw_abscissa='L2_validation',
                            hue='train_time',
                            hue_norm=(0.01, 0.15), legend=False)
# slicing
condition = lambda e: e <= 0.2
estim_hyper_param.slice(column='L2_validation', condition=condition, save=True)
histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
histplot_hyperparam.hist(column_name_draw='train_time', separators_plot=None, hue='reservoir size',
                         palette='RdYlBu', bins=20,
                         binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)

histplot_hyperparam.hist(column_name_draw='L2_validation', separators_plot=None, hue='reservoir size',
                         palette='RdYlBu', bins=20,
                         binrange=None, stat='count', multiple="dodge", kde=False, path_save_plot=None)

scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
scatplot_hyperparam.scatter(column_name_draw='L2_training', second_column_to_draw_abscissa='L2_validation',
                            hue='train_time', hue_norm=(0.01, 0.2), legend=False)

#################### finding the best model:
df_best = estim_hyper_param.get_best_by(metrics='L2_training', count=10)
print(df_best.to_string())
index_best = df_best.index[0]  # best model
path2net_best = linker_models([f"model_{index_best}.pth"])
path2estim_best = linker_estims([f"estim_{index_best}.json"])

config_architecture_second_elmt = lambda param: config_architecture(param, input_dim,
                                                                    output_dim)  # fetch only the class
best_model = create_model_by_index(index_best, PATH_JSON_PARAMS,
                                   path2net_best, config_architecture_second_elmt,
                                   mapping_names2functions=mapping_names2functions,
                                   flag_factory=False)

####### NO HISTORY IN OUR CASE.
# plotting history of this model
# estimator_history = Estim_history.from_json(path2estim_best, compressed=False)
# history_plot = corai.Relplot_history(estimator_history)
# history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot='L4', log_axis_for_loss=True,
#                                         log_axis_for_second_axis=True)
# history_plot.lineplot(log_axis_for_loss=True)

# wip input plot_xx and plot_yy
# plotting the prediction of this model


batch_nb_plot = 0
aplot = APlot(how=(1, 1))
plot_yy_pred = best_model.nn_predict_ans2cpu(inputs_test_scaled)
plot_prediction(inputs_test[0, washout + lookforward_window:, 0],
                minimax_out.inverse_transform(plot_yy_pred[:, batch_nb_plot]),
                minimax_out.inverse_transform(outputs_test_scaled[washout:, batch_nb_plot]),
                'Prediction over whole Set, descaled value.', aplot)

APlot.show_plot()
