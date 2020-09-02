#error specific to setters for classes, when the type is not the requested one.

class Error_type_setter(TypeError):
    def __init__(self, reason=''):
        self.message = " ".join(["Argument in setter is not of the good type. ", reason])
        super().__init__(self.message)
