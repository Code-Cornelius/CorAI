# type of error for casses of inputs which hasn't been coded yet.

class Error_not_yet_allowed(ValueError):
    def __init__(self, reason):
        self.message = " ".join(["Behaviour not allowed yet. Please wait. ", reason])
        super().__init__(self.message)
