class Error_convergence(Exception):
    def __init__(self, reason):
        self.message = " ".join(["Did not converge :", reason])
        super().__init__(self.message)