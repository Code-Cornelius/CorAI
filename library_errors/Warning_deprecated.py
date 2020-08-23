import warnings

def deprecated_function(reason="Unprecised."):
    message = " ".join(["Deprecated function name : ", reason])
    warnings.warn(message, DeprecationWarning)

