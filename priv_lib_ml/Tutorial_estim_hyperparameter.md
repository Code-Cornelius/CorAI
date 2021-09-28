# Hyper-parameters example

The tutorial follows closely the `tests/example_hyper_param.py` file.

### 0. Setup (Grid search)

The library offers an easy way to produce all the combinations of the parameters tweaked: a grid search regarding a list
of parameters.

- Define the options for each parameter as a dictionary of lists:

```python
params_options = {
    "lr": [0.0001, 0.001, 0.01, 0.1],
    'activation_function': ['tanh'],
    "dropout": [0., 0.2, 0.5],
    "list_hidden_sizes": [[2, 4, 2], [4, 8, 4]]
}
```

- Use the `parameter_product` function to produce all the possible combinations in the format of a list:

```python
hyper_params = parameter_product(params_options)
for params in hyper_params:
    pass  # code
```

- `params` is a dictionary containing the parameters that will change during different trainings and which will be
  compared for performance at the end.

### 1. During Training

The target is to keep the hyper-parameters used for training as close as possible to the history of the training. This
way, is it possible to evaluate the performances of a neural network and of its training.

Add the parameter `hyper_param` to the function `nn_kfold_train` where it is passed down to the
function `initialise_estimator`:

```python
net, estimator_history = nn_kfold_train(train_X, train_Y,
                                        model_class, params_training,
                                        early_stoppers,
                                        nb_split, shuffle_kfold, percent_val_for_1_fold,
                                        silent,
                                        hyper_param=param)
#################
# inside the function:
#################

estimator_history = initialise_estimator(compute_validation, param_train,
                                         train_param_dict=param)
```

This way, the hyper-parameters used are stored as meta-data in the estimator.

#### Unique ID for identification

By using a unique id, one can link by the name the results of the training (the saved net) and the hyper-parameters
used.

One can proceed in the following way:

1. Keeping an index to be used as an id while iterating over the `hyper_params`,

```python
for id, params in enumerate(hyper_params):
    pass  # code
```

2. Adding the id to the `params` dictionary,

```python
params['id'] = id
```

3. Using the id as part of the filename when saving the net.

```python
file_name = f'some_name_{id}'
net.save_net(path=os.path.join(some_path, file_name))
```

### 2. Creating an `Estim_hyper_param`

An `estim_hyper_param` is an estimator used to store information related to different training configurations and their
performance. The columns of the estimator are the names of the parameters we want to follow, the metrics we choose for
comparison and, optionally, the training time. Each line of the estimator will represent a configuration.

The library offers three ways to create an `Estim_hyper_param`:

#### 2.a From a list

One can create an `Estim_hyper_param` from an existing list of `Estim_history` estimators. Following the example above,
one could've appended every `estim_history` to a list after each training.

- Using the list `estims`:

```python
estim_hyper_param = Estim_hyper_param.from_list(estims, metric_names, flg_time)
```

- If the flat flg_time is set to `True` the training times will also be stored besides any metric results.

#### 2.b From a folder

If instead of appending all of `estim_history` estimators to a list one decides to save them individually as **json**
files, all in the same folder, they can then be retrieved using the path:

```python
estim_hyper_param = Estim_hyper_param.from_folder(path, metric_names, flg_time, compressed)
```

#### 2.c From a json/csv file

If the `estim_hyper_param` was previously stored as a json/csv file it can be retrieved using the path.

```python
estim_hyper_param = Estim_hyper_param.from_csv(path)
# or
estim_hyper_param = Estim_hyper_param.from_json(path)

```

### Slicing

The resulting estimator can be sliced. That way, it is possible to visualise only part of the data with the plotters. In
order to slice one needs to first define the slicing condition:

```python
condition = lambda column: column <= 2
```

The we can use the `slice` function to slice the estimator by applying the condition defined above to a column.

```python
sliced_datafram = estim_hyper_param.slice(column, condition)
```

We can also update the estimator if we want the change to be inplace.

```python
estim_hyper_param.slice(column, condition, save=True)
```

### Next steps:

- The `net` can be saved to file for later use. For more details
  check: `priv_lib_ml/src/classes/architecture/how_to_new_architectures.md`

- For more details on estimators check: `priv_lib_estimator/how_to_use_estimators_and_plotters.md`.