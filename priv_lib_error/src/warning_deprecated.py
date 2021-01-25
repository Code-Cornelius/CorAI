import warnings


def deprecated_function(reason="Not Specified."):
    """
    SEMANTICS :
        call that function whenever the function used is deprecated.

    Args:
        reason: additional information for the warning.
    """
    message = " ".join(["Deprecated function name : ", reason])
    warnings.warn(message, DeprecationWarning)
