class Error_not_allowed_input(ValueError):
    """
        type of error when I don't allow certain behaviours, in particular, when I don't accept some inputs.
    """
    DEFAULT_MESSAGE = "Values for the inputs not allowed. Please change the values. "

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join(
                [Error_not_allowed_input.DEFAULT_MESSAGE, args[0]])
        else:
            self.message = Error_not_allowed_input.DEFAULT_MESSAGE
        super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        if self.message:
            return self.message
        else:
            return Error_not_allowed_input.DEFAULT_MESSAGE
