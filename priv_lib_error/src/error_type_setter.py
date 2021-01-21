class Error_type_setter(TypeError):
    """
        SEMANTICS : error specific to setters for classes, when the type is not the requested one.
    """
    DEFAULT_MESSAGE = "Argument in setter is not of the good type. "

    def __init__(self, *args, **kwargs):
        if args:
            self.message = " ".join([Error_type_setter.DEFAULT_MESSAGE, args[0]])
        else:
            self.message = Error_type_setter.DEFAULT_MESSAGE
        super().__init__(self.message, *args, **kwargs)

    def __str__(self):
        return self.message

