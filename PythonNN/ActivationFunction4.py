import numpy as np

class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 2 / (1 + np.exp(- 2 * inputs)) - 1
        self.useful_backprop = self.output

    def backprop(self, gradient):
        self.inputs_deriv = gradient.copy()
        self.inputs_deriv = 1 - self.useful_backprop * self.useful_backprop