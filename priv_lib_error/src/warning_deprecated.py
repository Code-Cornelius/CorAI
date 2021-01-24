import warnings

def deprecated_function(reason="Not Specified."):
    """
    SEMANTICS : call that function whenever I am using an old version function.

    Args:
        reason:

    Returns:

    """
    message = " ".join(["Deprecated function name : ", reason])
    warnings.warn(message, DeprecationWarning)
