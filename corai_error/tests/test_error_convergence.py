# normal libraries
import unittest

# priv_libraries
from corai_error import Error_convergence

from corai_error.tests.broken_function import broken_function_with_message, broken_function_without_message


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Test_Error_convergence(unittest.TestCase):
    def test_Error_convergence_with_message(self):
        a_message = broken_function_with_message(Error_convergence)
        DEFAULT = Error_convergence.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + " A message ?All goodI continue running !")

    def test_Error_convergence_without_message(self):
        a_message = broken_function_without_message(Error_convergence)
        DEFAULT = Error_convergence.DEFAULT_MESSAGE
        assert (a_message == DEFAULT + "All goodI continue running !")
