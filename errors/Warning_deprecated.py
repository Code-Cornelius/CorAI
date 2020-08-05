import warnings

def deprecated_function(reason="No one"):
    message = " ".join(["Deprecated function name : ", reason])
    warnings.warn(message, DeprecationWarning)

