class Error_convergence(Exception):

    # see if you need more info
    def __init__(self, reason):
        self.message = "Did not converge -> " + reason
        super().__init__(self.message)

