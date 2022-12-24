from ActivationFunction1 import Activation_ReLU
from Layer import Layer_Dense

class MLP:
    def __init__(self):
        self.input_layer = Layer_Dense(9, 4)
        self.output_layer = Layer_Dense(4, 2)
        self.activ1 = Activation_ReLU()
    def forward(self, inputs):
        self.input_layer.forward(inputs)
        Z1 = self.input_layer.output
        self.activ1.forward(Z1)
        A1 = self.activ1.output
        self.output_layer.forward(A1)
        Z2 = A2 = self.output_layer.output
        return Z1, A1, Z2, A2
    def back_prop(self, Z1, A1, Z2, A2, X, Y):
        # Loss is (y_pred - y)^2
        # A2 is y_pred. Y is y
        dloss_dypred = 2 * (A2 - Y)
        # dypred_dz3 = A2 * (1 - A2)
        dypred_dz3 = 1
        dz3_dw2 = A1
        dz3_db2 = 1
        dz3_da2 = self.output_layer.weights # w2
        da2_dz2 = A1 * (1 - A1)
        dz2_dw1 = X
        dz2_db1 = 1
        
        dW1 = dloss_dypred * dypred_dz3 * dz3_da2 * da2_dz2 * dz2_dw1
        db1 = dloss_dypred * dypred_dz3 * dz3_da2 * da2_dz2 * dz2_db1
        dW2 = dloss_dypred * dypred_dz3 * dz3_dw2
        db2 = dloss_dypred * dypred_dz3 * dz3_db2
        return dW1, db1, dW2, db2
    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2
    def gradient_descent(self, X, Y, iterations, alpha):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward(X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, Y)
            W1, b1, W2, b2 = update_params(
                self.input_layer.weights,
                self.input_layer.biases,
                self.output_layer.weights,
                self.output_layer.biases,
                dW1, db1, dW2, db2, alpha)
            if i % 50 == 0:
                print("Iteration: ", i)
        return W1, b1, W2, b2