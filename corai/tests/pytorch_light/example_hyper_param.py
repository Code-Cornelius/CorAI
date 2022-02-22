import time

import numpy as np
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm

import corai
from config import ROOT_DIR
from corai.src.classes.pl.history_dict import History_dict
from corai.src.classes.pl.progressbar_without_val_batch_update import \
    Progressbar_without_val_batch_update
from corai.tests.pytorch_light.example_sinus_no_comments import Sinus_model, MyDataModule
from corai.tests.sinus_dataset_generator import data_sinus
from corai_plot import APlot
from corai_util.tools.src import function_dict, function_writer
from corai_util.tools.src.function_writer import factory_fct_linked_path

# we reuse the classes from the example of sinus_no_comments.
############################## GLOBAL PARAMETERS
NEW_DATASET = True
SAVE_TO_FILE = True

############################# paths definition
path_linker = factory_fct_linked_path(ROOT_DIR, 'corai/tests/pytorch_light/')
PATH_JSON_PARAMS = path_linker(["out", "param_hyper_param_tuning.json"])
linker_estims = function_writer.factory_fct_linked_path(ROOT_DIR, 'corai/tests/pytorch_light/estims_hp_opt')
linker_models = function_writer.factory_fct_linked_path(ROOT_DIR, 'corai/tests/pytorch_light/model_hp_opt')

seed_everything(42, workers=True)

############################# DATA CREATION
train_X, train_Y, testing_X, testing_Y, plot_xx, plot_yy, plot_yy_noisy, xx, yy = data_sinus()

############################# grid search arguments
params_options = {
    "architecture": ["fcnn"],
    "seed": [42, 124],
    "lr": [0.01, 0.1],
    'activation_function': ['tanh', 'relu'],
    "dropout": [0.2, 0.5],
    "list_hidden_sizes": [[2, 4, 2], [4, 8, 4], [2, 32, 2], [32, 128, 32]],
}
# convert parameters to the product of the parameters
hyper_params = function_dict.parameter_product(params_options)

# save the parameters
function_writer.list_of_dicts_to_json(hyper_params, file_name=PATH_JSON_PARAMS)
print(f"File {PATH_JSON_PARAMS} has been updated.")
print(f"    Number of configurations: {len(hyper_params)}.")

mapping_names2functions = {'tanh': torch.tanh, 'celu': torch.tanh, 'relu': torch.relu}


def config_architecture(params):
    params = params.copy()
    function_dict.replace_function_names_to_functions(params, mapping_names2functions, silent=True)

    # config of the architecture:
    input_size = 1
    hidden_sizes = params["list_hidden_sizes"]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [params['activation_function']] * 3
    # activation_functions = [torch.tanh, torch.tanh, torch.tanh]
    # activation_functions = [torch.celu, torch.celu, torch.celu]
    # activation_functions = [torch.relu, torch.relu, torch.relu]

    dropout = params["dropout"]
    ############################### Init our model
    sinus_model = Sinus_model(input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                              lr=params["lr"], weight_decay=0.00001, aplot_flag=False)

    return sinus_model


def config_trainer():
    ############################### Init the Early Stopper
    period_log = 20
    early_stop_val_loss = EarlyStopping(monitor="val_loss", min_delta=1E-3, patience=100 // period_log,
                                        verbose=False, mode="min", )
    ############################### Init the history and checkpoints
    logger_custom = History_dict(metrics=["val_loss", "train_loss"], aplot_flag=False,
                                 frequency_epoch_logging=period_log, )
    chckpnt = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False, save_top_k=1,
                              dirpath=linker_models(['']))
    ############################### Init the trainer and data
    AVAIL_GPUS = 0
    trainer = Trainer(default_root_dir=path_linker(['out']),
                      gpus=AVAIL_GPUS, max_epochs=7500,
                      logger=[logger_custom],
                      check_val_every_n_epoch=period_log,
                      num_sanity_val_steps=0,
                      callbacks=[early_stop_val_loss, Progressbar_without_val_batch_update(refresh_rate=10),
                                 chckpnt, ])
    return trainer, logger_custom, chckpnt


def generate_estims_history(hyper_params, config_architecture):
    estims = []
    for i, params in enumerate(tqdm(hyper_params)):
        sinus_model = config_architecture(params)

        BATCH_SIZE = 200000
        sinus_data = MyDataModule(xx, yy, BATCH_SIZE)
        trainer, logger_custom, chckpnt = config_trainer()
        ############################### Training
        start_time = time.perf_counter()
        trainer.fit(sinus_model, datamodule=sinus_data)
        final_time = time.perf_counter() - start_time
        train_time = np.round(time.perf_counter() - start_time, 2)
        print("Total time training: ", train_time, " seconds. In average, it took: ",
              np.round(train_time / trainer.current_epoch, 4), " seconds per epochs.")
        if SAVE_TO_FILE:
            estimator_history = logger_custom.to_estim_history(checkpoint=chckpnt, train_time=final_time)
            estimator_history.to_json(linker_estims([f'estim_{i}.json']), compress=False)
            # TODO WE WANT TO BE ABLE TO CHOSE WHERE THE CHECKPT IS LOCATED. IN EXAMPLE_HP and in the SINUS_EXAMPLE.
            estims.append(estimator_history)

    return estims


if __name__ == '__main__':

    if NEW_DATASET:
        print("Training.")
        estims = generate_estims_history(hyper_params, config_architecture)
    print("Plotting.")
    estim_hyper_param = corai.Estim_hyper_param.from_folder(linker_estims(['']),
                                                            metric_names=["loss_validation", "loss_training"],
                                                            flg_time=True, compressed=False)

    ######## example of usage:
    # estim_hyper_param = corai.Estim_hyper_param.from_list(estims, metric_names=["loss_validation", "loss_training"],
    #                                                 flg_time=True)
    # estim_hyper_param.to_csv("other_csv_from_examples/example_estim_hyper_param.csv")
    # estim_hyper_param.compute_number_params_for_fcnn()

    ######## drawing the distribution plot:
    histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
    histplot_hyperparam.hist(column_name_draw='train_time', hue='dropout',
                             separators_plot=None,
                             palette='RdYlBu', bins=50,
                             binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)
    histplot_hyperparam.hist(column_name_draw='loss_validation',
                             separators_plot=None,
                             hue='lr',
                             palette='RdYlBu', bins=20,
                             binrange=None, stat='count', multiple="dodge", kde=True, path_save_plot=None)

    ######## drawing the relation plot:
    scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
    scatplot_hyperparam.scatter(column_name_draw='loss_training',
                                second_column_to_draw_abscissa='loss_validation',
                                hue='train_time',
                                hue_norm=(0, 30), legend=False)

    ######## conditioning the data to plot a subset:
    condition = lambda t: t <= 2.
    estim_hyper_param.slice(column='train_time', condition=condition, save=True)  # slice data, removing some part.

    histplot_hyperparam = corai.Distplot_hyper_param(estimator=estim_hyper_param)
    histplot_hyperparam.hist(column_name_draw='train_time',
                             separators_plot=None,
                             hue='dropout',
                             palette='RdYlBu', bins=50,
                             binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)

    histplot_hyperparam.hist(column_name_draw='loss_validation',
                             separators_plot=None,
                             hue='lr',
                             palette='RdYlBu', bins=20,
                             binrange=None, stat='count', multiple="dodge", kde=True, path_save_plot=None)

    scatplot_hyperparam = corai.Relplot_hyper_param(estimator=estim_hyper_param)
    scatplot_hyperparam.scatter(column_name_draw='loss_training', second_column_to_draw_abscissa='loss_validation',
                                hue='train_time', hue_norm=(0, 2), legend=False)

    # TODO 22/02/2022 nie_k:  finish this
    #################### finding the best model:
    # df_best = estim_hyper_param.get_best_by(metrics='loss_validation', count=3)
    # print(df_best.to_string())
    # index_best = df_best.index[0]  # best model
    # path2net_best = linker_models([f"model_{index_best}.pth"])
    # path2estim_best = linker_estims([f"estim_{index_best}.json"])
    #
    # config_architecture_second_elmt = lambda param: config_architecture(param)[1]  # fetch only the class
    # best_model = create_model_by_index(index_best, PATH_JSON_PARAMS,
    #                                    path2net_best, config_architecture_second_elmt,
    #                                    mapping_names2functions=mapping_names2functions,
    #                                    flag_factory=True)
    #
    # # plotting history of this model
    # estimator_history = Estim_history.from_json(path2estim_best, compressed=False)
    # history_plot = corai.Relplot_history(estimator_history)
    # history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot='L4', log_axis_for_loss=True,
    #                                         log_axis_for_second_axis=True)
    # history_plot.lineplot(log_axis_for_loss=True)
    #
    # # plotting the prediciton of this model
    # corai.nn_plot_prediction_vs_true(net=best_model, plot_xx=plot_xx, plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy)

    APlot.show_plot()
