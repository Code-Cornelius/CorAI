# normal libraries
import unittest

# my libraries
from priv_lib_error import Error_type_setter

from priv_lib_error.tests.broken_function import broken_function_with_message, broken_function_without_message


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Test_Error_type_setter(unittest.TestCase):

    def test_Error_type_setter_input_with_message(self):
        a_message = broken_function_with_message(Error_type_setter)
        DEFAULT = Error_type_setter.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + " A message ?All goodI continue running !")

    def test_Error_type_setter_without_message(self):
        a_message = broken_function_without_message(Error_type_setter)
        DEFAULT = Error_type_setter.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + "All goodI continue running !")

