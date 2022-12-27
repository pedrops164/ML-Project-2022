from ActivationFunction1 import Activation_ReLU
from Layer import Layer_Dense
import numpy as np
from LossMeanSquare import LossMeanSquare
from ActivationLoss1 import Activation_Linear_Loss_LMS

class NN1:
    def __init__(self):
        self.layer = Layer_Dense(9,2)
        self.activ_loss = Activation_Linear_Loss_LMS()
    def forward(self, inputs, Y):
        self.layer.forward(inputs)
        Z1 = self.layer.output
        loss = self.activ_loss.forward(Z1, Y)
        return loss
    def back_prop(self, Y):
        n_samples = Y.shape[0] # number of examples

        self.activ_loss.backprop(Y)
        self.layer.backprop(self.activ_loss.inputs_deriv)

    def update_params(self, alpha):
        self.layer.update_params(alpha)

    def gradient_descent(self, X, Y, iterations, alpha):
        for i in range(iterations):
            loss = self.forward(X, Y)
            self.back_prop(Y)
            self.update_params(alpha)
            if i % 5 == 0:
                print("Iteration: ", i)
                print(loss)
