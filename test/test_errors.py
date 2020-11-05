# normal libraries
import unittest


# my libraries
from library_errors import error_type_setter, error_not_allowed_input, error_not_enough_information, error_convergence, error_not_yet_allowed

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Test_error_raising(unittest.TestCase):
    def test_error_convergence(self):
        # we expect : "A message ?", then "All good" and "I continue running!".
        def broken_function(a_message):
            try:
                raise error_convergence.Error_convergence("A message ?")

            except error_convergence.Error_convergence as e:
                a_message += "A message ?"
                print(e)
            finally:
                a_message += "All good"
                print("All good")
            a_message += "I continue running !"
            print("I continue running !")
            return a_message

        a_message = ""
        a_message = broken_function(a_message)
        print(a_message)
        assert(a_message == "A message ?All goodI continue running !")



    def test_error_type_setter(self):
        # we expect : "All good" and no error handling.
        def broken_function():
            try:
                raise error_convergence.Error_convergence("A message ?")
            except error_not_yet_allowed.Error_not_yet_allowed as e:
                print(e)
            finally:
                print("All good")
            print("I continue running !")

        with self.assertRaises(error_convergence.Error_convergence) as context:
            broken_function()

        self.assertTrue("A message ?" in str(context.exception))
