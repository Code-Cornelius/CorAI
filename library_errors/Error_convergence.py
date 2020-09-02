# type of error when an algorithm didn't converged to a satisfying answer.

class Error_convergence(Exception):
    def __init__(self, reason):
        self.message = " ".join(["Did not converge :", reason])
        super().__init__(self.message)
