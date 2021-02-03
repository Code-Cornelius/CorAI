class Error_not_allowed_input(ValueError):
    """
        SEMANTICS : type of error for when a certain behaviour is not allowed,
                    in particular, when some inputs is not allowed.
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
        return self.message
