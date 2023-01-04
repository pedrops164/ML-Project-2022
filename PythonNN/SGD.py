
# Stochastic Gradient Descent
class SGD:

    def __init__(self, learning_rate=0.7, decay=0., momentum=0., min_lr = 0., lambda_param=0.):
        self.starting_learning_rate = learning_rate
        self.lr_decay = decay
        self.lr = self.starting_learning_rate
        self.n_iterations = 0
        self.momentum = momentum
        self.min_lr = min_lr
        self.lambda_param = lambda_param

    def decay_lr(self):
        if self.lr_decay != 0 and self.min_lr < self.lr:
            # we update the learning rate, if the learning rate decay is not zero,
            # and if we haven't reached the minimum learning rate
            self.lr = self.lr * (1. / (1. + self.lr_decay))

    def adjust_parameters(self, layer):

        weight_changes = self.momentum * layer.momentums_weight \
            -self.lr * layer.weights_deriv - 2 * self.lambda_param * layer.weights
        layer.momentums_weight = weight_changes

        bias_changes = self.momentum * layer.momentums_bias \
            -self.lr * layer.biases_deriv
        layer.momentums_bias = bias_changes

        # update weight and biases with the variation
        layer.weights += weight_changes
        layer.biases += bias_changes

    def increase_iteration(self):
        self.n_iterations += 1



