from ActivationFunction1 import Activation_ReLU
from Layer import Layer_Dense
import numpy as np
from LossMeanSquare import LossMeanSquare
from ActivationLoss1 import Activation_Linear_Loss_LMS
import matplotlib.pyplot as plt

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

    def gradient_descent(self, X, Y, test_X, test_Y, iterations, alpha):
        train_size = range(iterations)
        train_Y_data = [] # loss
        test_Y_data = [] # loss
        for i in range(iterations):
            loss_validation = self.forward(test_X, test_Y)
            test_Y_data.append(np.mean(loss_validation))
            loss_empirical = self.forward(X, Y)
            self.back_prop(Y)
            self.update_params(alpha)
            train_Y_data.append(np.mean(loss_empirical))
            if i % 5 == 0:
                print("Iteration: ", i)
                print(loss_empirical)
                print(loss_validation)

        plt.plot(train_size, train_Y_data, '--', color="#111111", label="Training loss")
        plt.plot(train_size, test_Y_data, color="#111111", label="Validation loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title("Learning curves")
        plt.axis([0,iterations,0,20])
        plt.show()
