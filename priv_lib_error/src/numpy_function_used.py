import warnings


def numpy_function_used(which_function_used_instead="Not Specified."):
    """
    Semantics:
        Whenever a numpy function should be used instead, triggers a warning.
    """
    message = " ".join(
        ["The object given is not a list, but an array. The numpy function ", which_function_used_instead, " is used."])
    warnings.warn(message, DeprecationWarning)
