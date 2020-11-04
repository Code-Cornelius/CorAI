class Error_convergence(Exception):
    """
        type of error when an algorithm didn't converged to a satisfying answer.
    """
    DEFAULT_MESSAGE = 'Error of convergence. '

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join([Error_convergence.DEFAULT_MESSAGE, args[0]])
        else:
            super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        if self.message:
            return self.message
        else:
            return Error_convergence.DEFAULT_MESSAGE
