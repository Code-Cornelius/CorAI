class Error_not_yet_allowed(ValueError):
    """
       SEMANTICS : type of error for cases of inputs which haven't been implemented yet.
    """
    DEFAULT_MESSAGE = "Behaviour not allowed yet. Please await new version. "

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join([Error_not_yet_allowed.DEFAULT_MESSAGE, args[0]])
        else:
            self.message = Error_not_yet_allowed.DEFAULT_MESSAGE
        super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        return self.message
