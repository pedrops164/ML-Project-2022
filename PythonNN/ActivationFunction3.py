class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backprop(self, gradient):
        # the derivative of the linear activation function is 1,
        # therefore the derivative doesnt change
        self.inputs_deriv = gradient.copy()