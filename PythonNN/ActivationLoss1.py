from ActivationFunction3 import Activation_Linear
from LossMeanSquare import LossMeanSquare

class Activation_Linear_Loss_LMS():
    def __init__(self):
        self.activation_func = Activation_Linear()
        self.loss_func = LossMeanSquare()

    def forward(self, inputs, Y):
        self.activation_func.forward(inputs)
        self.output = self.activation_func.output

        return self.loss_func.calculate(self.output, Y)

    def backprop(self, Y):
        n_samples = Y.shape[0]
        loss_gradient = 2 * (self.output - Y)
        self.inputs_deriv = loss_gradient / n_samples

