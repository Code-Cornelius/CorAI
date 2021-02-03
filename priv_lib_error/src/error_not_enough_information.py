class Error_not_enough_information(ValueError):
    """
        SEMANTICS : type of error for when certain behaviours are not allowed,
                    in particular, when some inputs are not allowed.
    """
    DEFAULT_MESSAGE = "Behaviour not allowed yet. Please recheck. You need to give more input/information. "

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join([Error_not_enough_information.DEFAULT_MESSAGE, args[0]])
        else:
            self.message = Error_not_enough_information.DEFAULT_MESSAGE
        super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        return self.message
