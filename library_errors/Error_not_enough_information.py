# type of error when I don't allow certain behaviours, in particular, when I don't accept some inputs.

class Error_not_enough_information(ValueError):
    def __init__(self, reason):
        self.message = " ".join(["Behaviour not allowed yet. Please recheck. ", reason])
        super().__init__(self.message)
