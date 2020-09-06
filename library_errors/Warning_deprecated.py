import warnings


# call that function whenever I am using an old version function.

def deprecated_function(reason="Not Specified."):
    message = " ".join(["Deprecated function name : ", reason])
    warnings.warn(message, DeprecationWarning)
