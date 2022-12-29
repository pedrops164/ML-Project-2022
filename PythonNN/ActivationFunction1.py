import numpy as np

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