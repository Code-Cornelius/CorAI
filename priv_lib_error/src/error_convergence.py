class Error_convergence(Exception):
    """
        Semantics:
            Type of error for when an algorithm didn't converge to a satisfying answer.
    """
    DEFAULT_MESSAGE = 'Error of convergence. '

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join([Error_convergence.DEFAULT_MESSAGE, args[0]])
        else:
            self.message = Error_convergence.DEFAULT_MESSAGE
        super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        return self.message
