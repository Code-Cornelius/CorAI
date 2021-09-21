`Savable_net` is the basis of nets, giving useful methods in order to save a neural network at the right time. 
`Savable_net` requires the parameter `predict_fct` to its init. If None is given, a default `predict_fct` is used.

# Example of derived class from `Savable_net`:

```python
class NN(Savable_net, metaclass=ABCMeta):
    def __init__(self, predict_fct, *args, **kwargs):
        super().__init__(predict_fct, *args, **kwargs)

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, new_parameter):
        if isinstance(new_parameter, int):
                self._parameter = new_parameter
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    def forward(self, x):
        return x * parameter


def factory_parametrised_NN(parameter, param_predict_fct):
    class Parametrised_FC_NN(Fully_connected_NN):
        # defining attributes this way shadows the abstract properties from parents.
        parameter = param_parameter

        def __init__(self):
            super().__init__(predict_fct=param_predict_fct)
            # :to initialize all the layers and dropout with respect to the parameters created.

    return Parametrised_NN
```
For all neural networks architecture, we use a construction in two pieces. First the actual class, then the class factory. 
We use class factory because since the basic of k-fold is creating different neural networks that are trained on different data.

It also allows to have the class at hand when one wants to use hyper-parameters estimator.

# Training:

One can use the helper `NNTrainParameters` that stores all the important hyper-parameters for training.

Then, the usual method to use is:

```python
nn_kfold_train(data_training_X, data_training_Y, model_NN, parameters_training,
                early_stopper_validation=Early_stopper_vanilla(), early_stopper_training=Early_stopper_vanilla(), nb_split=5,
                shuffle_kfold=True, percent_validation_for_1_fold=20, compute_accuracy=False, silent=False)
```


### todo

talk about not using kfold as in forecast flight.


talk about estimator history and estimator hyperparam.