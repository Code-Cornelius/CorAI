# normal libraries
import unittest

# priv_libraries
from corai_error import Error_not_allowed_input

from corai_error.tests.broken_function import broken_function_with_message, broken_function_without_message


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Test_Error_not_allowed_input(unittest.TestCase):
    def test_Error_not_allowed_input_with_message(self):
        a_message = broken_function_with_message(Error_not_allowed_input)
        DEFAULT = Error_not_allowed_input.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + " A message ?All goodI continue running !")

    def test_Error_not_allowed_input_without_message(self):
        a_message = broken_function_without_message(Error_not_allowed_input)
        DEFAULT = Error_not_allowed_input.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + "All goodI continue running !")
