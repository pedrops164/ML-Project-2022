import numpy as np

class Sigmoid():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backprop(self, gradient):
        self.inputs_deriv = gradient * self.output * (1 - self.output)