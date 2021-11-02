from typing import Callable

from priv_lib_error import Error_type_setter


class Metric(object):
    """
    Metric wrapper.
    Examples
        def L4loss(net,xx,yy):
            return torch.norm(net.nn_predict(xx) - yy, 4)
        L4metric = Metric('L4',L4loss)
        metrics = (L4metric,)

    """

    def __init__(self, name, function):
        """
        Constructor.
        Args:
            name: a name that should be used to refer to the metric
            function: a callable taking 3 parameters: net, xx and yy. It returns a float.
            Be careful about how the data is computed,
            as net is on device, xx is on device, and yy is also on device.
        """
        self.name = name
        self._function = function

    def __call__(self, net, xx, yy):
        return self._function(net, xx, yy)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise Error_type_setter(f'Argument is not an {str(str)}.')

    @property
    def _function(self):
        return self.__function

    @_function.setter
    def _function(self, new__function):
        if isinstance(new__function, Callable):
            self.__function = new__function
        else:
            raise Error_type_setter(f'Argument is not an {str(Callable)}.')
