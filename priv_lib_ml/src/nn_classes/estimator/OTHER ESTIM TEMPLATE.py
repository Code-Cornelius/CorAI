# PARAMETERS
number_of_diff_configs = 972  # length JSON.
percent_best = 0.02  # 1% for 1000 is 10
name_config = 'type1'

# data:
L2_training = [0] * number_of_diff_configs
L4_training = [0] * number_of_diff_configs
L2_validation = [0] * number_of_diff_configs
L4_validation = [0] * number_of_diff_configs

for index in tqdm(range(1, number_of_diff_configs + 1)):
    path_to_history = linked_path_result(['history', name_config, f"history_{index}.json"])
    with open(path_to_history, 'r') as file:
        history = json.load(file)
        history = unzip_json(history)
        file.close()
    best_epoch = history['best_epoch'][0]  # works ?
    L2_training[index - 1] = history['training']['loss'][0][best_epoch]
    L4_training[index - 1] = history['training']['L4'][0][best_epoch]
    L2_validation[index - 1] = history['validation']['loss'][0][best_epoch]
    L4_validation[index - 1] = history['validation']['L4'][0][best_epoch]

scatter = APlot()
scatter.uni_plot(nb_ax=0, xx=L2_validation, yy=L4_validation,
                 dict_plot_param={'color': 'red', 'linestyle': '', 'linewidth': 0,
                                  'markersize': 3, 'marker': 'x'},
                 dict_ax={'xlabel': 'L2 validation loss', 'ylabel': 'L4 validation loss',
                          'xscale': 'log', 'yscale': 'log', 'title': 'Comparison Different Trainings'})

dict_df = {'L2_training': L2_training, 'L4_training': L4_training, 'L2_validation': L2_validation,
           'L4_validation': L4_validation}
df = pd.DataFrame(dict_df)
df = df.sort_values(by="L2_validation", ascending=True)
print(df.loc[144])
df_best = df[:int(number_of_diff_configs * percent_best)]  # 10%
print(df_best.to_string())
path_scatter_plot = linked_path_result([f"history_sum_up_{name_config}"])
scatter.save_plot(path_scatter_plot)

# retrieve data with index given by df_best.
for index in tqdm(df_best.index):
    parameters = retrieve_parameters_by_index_from_json(index, linked_path_data(['parameters_grid_search.json']))
    print(f"for config {index}, the best parameters are : \n{parameters}")
    replace_function_names_to_functions(parameters, MAPPING, silent=True)

    (SEED, HIDDEN_SIZES, PARTICULAR_HIDDEN_LAYERS,
     ACTIVATION_FUNCTIONS, BATCH_SIZE, DROPOUT,
     OPTIMISER, DICT_OPTIMISER) = list(parameters.values())
