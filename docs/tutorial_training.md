# Training example

The tutorial follows closely the `example_hyper_param.py` file.

### 0. Good practices:

- import the libraries as `corai`:

```python
import corai
```

- get the device used for training:

```python
device = corai.pytorch_device_setting('cpu')
```

- set the seed:

```python
corai.set_seeds(seed)
```

- use the `decorator_train_disable_no_grad` decorator for prediction function helpers:

```python
@corai.decorator_train_disable_no_grad
def predict(data):
    pass
```

The decorator will disable the gradient computation. Since we don't want to perform backpropagation at this step,
disabling the gradient computation will reduce memory consumption. On top of that, the decorator puts the net in test
mode (`train(mode=False)`), which will disable the dropout and normalization. This is desired since we don't want to
affect the performance of the trained net during testing.

- instead of using the model directly for prediction, use the dedicated `nn_predict` function:

```python
prediction = net.nn_predict(data_to_predict)
```

The `nn_predict` function is wrapped by the `decorator_train_disable_no_grad`. Internally, `nn_predict`
calls the `prediction` function, in turn calling `self.predict_ftc`, which is passed to the `net` at instantiation.

### 1. Define the training parameters

The training parameters are passed with `NNTrainParameters`.

- initialise a `NNTrainParameters`

```python
params_training = corai.NNTrainParameters(batch_size, epochs,
                                          device, criterion,
                                          optim_wrapper, metrics)
```

In particular, one decides the metrics used to compute the loss during training. These are defined in the following way:

- define the `metric`s, metrics are always computed on the device (since net is):

```python
from torch import linalg as LA
def L4loss(net, xx, yy):
  return torch.norm(net.nn_predict(xx) - yy, 4)


L4metric = corai.Metric('L4', L4loss)
metrics = (L4metric,)  # tuple of metrics
```

One needs to also define how the optimisation step is performed. This is specified through the `Optim_wrapper`
parameter.

- define the `Optim_wrapper`:

```python
optimiser = torch.optim.Adam
optimiser_parameters = {"lr": 0.001,
                        "weight_decay": 0.00001}

optim_wrapper = corai.Optim_wrapper(optimiser, optimiser_parameters)
```

- one can also use a `scheduler` to adapt the learning rate during training:

```python
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_parameters = {"gamma": 0.1}

optim_wrapper = corai.Optim_wrapper(optimiser,
                              optimiser_parameters,
                              scheduler,
                              scheduler_parameters)
```

- define the `early_stoppers`:

```python
early_stop_train = corai.Early_stopper_training(patience=20, silent=SILENT, delta=-1E-6)
early_stop_valid = corai.Early_stopper_validation(patience=20, silent=SILENT, delta=-1E-6)
early_stoppers = (early_stop_train, early_stop_valid)
```

### 2. Configure the model class

- configure the class of the model that will be used for training:

```python
model_class = corai.factory_parametrised_FC_NN(param_input_size,
                                         param_list_hidden_sizes,
                                         param_output_size,
                                         param_list_biases,
                                         param_activation_functions,
                                         param_dropout,
                                         param_predict_fct)
```

### 3. Train

One trains the model using either k-fold training or not. The choice is made depending on how one wants to deal with the
data. The split in the k-fold is done randomly. In some cases, such splitting does not make sense. For example if one
wants to choose the distribution of the training data, one should not use `nn_kfold_train`, but split the data oneself
and use `train_kfold_a_fold_after_split`.

#### 3.a Training with k-fold

- train the model by K-Fold:

```python
net, estimator_history = corai.nn_kfold_train(train_X,
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

It is also possible to use the training pipeline, without any predefined splitting (and perhaps do the splitting by hand
beforehand). In order to use the function without the splitting, one should use `train_kfold_a_fold_after_split`. The
function takes as an input an `estim_history`, where the values of the training are stored. The steps are:

- initialise estimator:

```python
estimator_history = corai.initialise_estimator(compute_validation, param_train)
```

- training:

```python
net, value_metric_for_best_net = corai.train_kfold_a_fold_after_split(train_X, train_Y,
                                                                indices_train, indices_valid,
                                                                model_class, params_training,
                                                                estimator_history, early_stoppers)
```

### Automatically reload models when parameters are stored in a json.

It is possible to directly recreate a model by simply specifying where to find the hyper-parameters of the models.
There is a tutorial about `parameter_product` that does create such json file in
[`corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md).

The function to recreate the model is `create_model_by_index` that lies in `util_train.py`.

### Next steps:

- The `net` can be saved to file for later use. For more details check:
  [`corai/src/classes/architecture/how_new_architectures.md`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/src/classes/architecture/how_new_architectures.md)
- The results can be used for plotting:
  [`corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md),
  [`corai/src/classes/estimator/history/tutorial_estim_history.md`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/src/classes/estimator/history/tutorial_estim_history.md)

