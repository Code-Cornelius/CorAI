# type of error when I don't allow certain behaviours, in particular, when I don't accept some inputs.

class Error_not_allowed_input(ValueError):
    def __init__(self, reason):
        self.message = " ".join(["Values for the inputs not allowed. Please change the values. ", reason])
        super().__init__(self.message)
