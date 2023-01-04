class RegularizationL2:
    def __init__(self, lambda_param):
        self.lambda_param = lambda_param

    def regularize(self, layer):
        layer.weights_deriv += 2 * self.lambda_param * layer.weights


