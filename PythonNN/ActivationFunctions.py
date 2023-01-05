import numpy as np

class Sigmoid():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backprop(self, gradient):
        self.inputs_deriv = gradient * self.output * (1 - self.output)


class Activation_ReLU:
    def forward(self, input):
        self.inputs = input
        self.output = np.maximum(0, input)

    def backprop(self, gradient):
        n_samples = gradient.shape[0]
        self.inputs_deriv = gradient.copy()
        # we are putting null value for the gradient where the input values are negative
        self.inputs_deriv[self.inputs <= 0] = 0
        self.inputs_deriv /= n_samples

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backprop(self, gradient):
        # the derivative of the linear activation function is 1,
        # therefore the derivative doesnt change
        self.inputs_deriv = gradient.copy()

class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 2 / (1 + np.exp(- 2 * inputs)) - 1
        self.useful_backprop = self.output

    def backprop(self, gradient):
        self.inputs_deriv = gradient.copy()
        self.inputs_deriv = 1 - self.useful_backprop * self.useful_backprop