class Error_forbidden(Exception):
    def __init__(self, reason):
        self.message = " ".join(["Behaviour not allowed. Please recheck. ", reason])
        super().__init__(self.message)
