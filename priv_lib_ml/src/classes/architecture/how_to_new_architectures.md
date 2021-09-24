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

## Use of `Savable_net`:
### 1. Saving a net

A good practice is to save the net with a `.pth` extension, as recommended:
(cf. https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html).

```python
net.save_net(path)
```

### 2. Loading a net
In order to do this we need to create a net of the right architecture. Using a different architecture from
the one used to save the net to file will throw an error.

The initialised object will be a net with untrained parameters. From here we can use the `load_net` function
to populate the net with the previously trained parameters.

```python
net.load_net(path)
```