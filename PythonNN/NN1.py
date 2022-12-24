from ActivationFunction1 import Activation_ReLU
from Layer import Layer_Dense
import numpy as np
from LossMeanSquare import LossMeanSquare

class NN1:
    def __init__(self):
        self.layer = Layer_Dense(9,2)
    def forward(self, inputs):
        self.layer.forward(inputs)
        Z1 = A1 = self.layer.output
        return Z1, A1
    def back_prop(self, Z1, A1, X, Y):
        n_samples = Y.shape[0] # number of examples
        # Loss is (y_pred - y)^2
        # A2 is y_pred. Y is y
        dloss_dypred = 2 * (A1 - Y) # shape (batch_size, n_output)
        da1_dz1 = 1
        dz1_dw1 = X # shape (batch_size, n_input)
        dz1_db1 = np.ones(n_samples) # shape = (n_output,1)
        # dW1 = dloss_dypred * da1_dz1 * dz1_dw1
        dW1 = 1/n_samples * dloss_dypred.T.dot(da1_dz1).dot(dz1_dw1).T #shape was (n_output, n_input). we want (n_input, n_output)
        # db1 = dloss_dypred * da1_dz1 * dz1_db1
        db1 = 1/n_samples * dloss_dypred.T.dot(da1_dz1).dot(dz1_db1).T
        return dW1, db1
    def update_params(self, W1, b1, dW1, db1, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1

        return W1, b1
    def gradient_descent(self, X, Y, iterations, alpha):
        loss1 = LossMeanSquare()
        for i in range(iterations):
            Z1, A1 = self.forward(X)
            dW1, db1 = self.back_prop(Z1, A1, X, Y)
            W1, b1 = self.update_params(
                self.layer.weights,
                self.layer.biases,
                dW1, db1, alpha)
            self.layer.weights = W1
            self.layer.biases = b1
            if i % 5 == 0:
                print("Iteration: ", i)
                print(loss1.calculate(A1, Y))
        return W1, b1