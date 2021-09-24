# Training example
The tutorial follows closely the `example_hyper_param.py` file.
### 0. Setup

Good practices:

- get the device used for training

```python
device = pytorch_device_setting('cpu')
```

- set the seed
```python
from priv_lib_ml.src.util_training import set_seeds
set_seeds(seed)
```
# TODO
- when predicting values using the functions... that are wrapped up with the decorator...
- other good practices?

### 1. Define the training parameters

The training parameters are passed with `NNTrainParameters`. 

- initialise a `NNTrainParameters`

```python
params_training = NNTrainParameters(batch_size,
                                    epochs,
                                    device,
                                    criterion,
                                    optim_wrapper,
                                    metrics)
```

In particular, one decides the metrics used to compute the loss during training.
These are defined in the following way:

- define the `metric`s
```python
def L4loss(net, xx, yy):
    return torch.norm(net.nn_predict(xx) - yy, 4)

L4metric = Metric('L4', L4loss)
metrics = (L4metric,)
```

One needs as well to define how the optimisation step is undergone.
This is specified through the `Optim_wrapper` parameter.

- define the `Optim_wrapper`
```python
optimiser = torch.optim.Adam
optimiser_parameters = {"lr": 0.001,
                        "weight_decay": 0.00001}

optim_wrapper = Optim_wrapper(optimiser, optimiser_parameters)
```

 - one can also use a `scheduler` to adapt the learning rate during training

```python
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_parameters = {"gamma":0.1}

optim_wrapper = Optim_wrapper(optimiser,
                              optimiser_parameters,
                              scheduler,
                              scheduler_parameters)
```


- define the `early_stoppers`
```python
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=-int(1E-6))
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=-int(1E-6))
early_stoppers = (early_stop_train, early_stop_valid)
```

### 2. Configure the model class
- configure the class of the model that will be used for training
```python
model_class = factory_parametrised_FC_NN(param_input_size,
                                         param_list_hidden_sizes,
                                         param_output_size, 
                                         param_list_biases,
                                         param_activation_functions,
                                         param_dropout,
                                         param_predict_fct)
```

### 3. Train

One trains the model using either k-fold training or not. 
The choice is made depending on how one wants to deal with the data.
The split in the k-fold is done randomly. 
In some cases, such splitting does not make sense. 
For example if one wants to choose the distribution of the training data, 
one should not use `nn_kfold_train`, 
but split the data oneself and use `train_kfold_a_fold_after_split`.

#### 3.a Training with k-fold

- train the model by K-Fold:

```python
net, estimator_history = nn_kfold_train(train_X,
                                        train_Y,
                                        model_class,
                                        params_training,
                                        early_stoppers,
                                        nb_split,
                                        shuffle_kfold,
                                        percent_val_for_1_fold,
                                        silent)
```

#### 3.b Training without k-fold

It is also possible to use the training pipeline, without any predefined splitting (and perhaps do the splitting by hand beforehand).
In order to use the function without the splitting,  one should use `train_kfold_a_fold_after_split`. 
The function takes as an input an `estim_history`, where the values of the training are stored. The steps are:

- initialise estimator:
```python
estimator_history = initialise_estimator(compute_validation, param_train)
```

- training:
```python
net, _ = train_kfold_a_fold_after_split(train_X,
                                        train_Y,
                                        indices_train, 
                                        indices_valid,
                                        model_class,
                                        params_training,
                                        estimator_history,
                                        early_stoppers)
```

### Next setps:
- The `net` can be saved to file for later use. For more details check:
  *priv_lib_ml/src/classes/architecture/how_to_new_architectures.md*
- The results can be used for plotting: <link file for plotting>
