from Layer import ConvolutionalLayer, ReshapeLayer, Dense
from ActivationFunctions import Sigmoid
from LossFunctions import BCE
from Adjuster import CNN_Optimizer
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

class CNN:
    def __init__(self):
        self.layers = [
            ConvolutionalLayer((1,28,28), 3, 5),
            Sigmoid(),
            ReshapeLayer((5,26,26), (1, 5*26*26)),
            Dense(5*26*26, 100, Sigmoid()),
            Dense(100,2, Sigmoid())            
        ]

        self.lr = 0.1

        self.loss = BCE()
        self.optimizer = CNN_Optimizer()

    def forward(self, X):
        output = X
        for l in self.layers:
            l.forward(output)
            output = l.output

        return output

    def backprop(self, Y):
        self.loss.backprop(Y)
        gradient = self.loss.inputs_deriv
        for l in reversed(self.layers):
            l.backprop(gradient)
            l.update(self.optimizer)
            gradient = l.inputs_deriv


    def train(self, X_train, Y_train, epochs):

        for it in range(epochs):
            error = 0
            for x,y in zip(X_train, Y_train):
                Y_pred = self.forward(x)
                error += self.loss.calculate(Y_pred, y)
                self.backprop(y)
            print("error", error / len(Y_pred))


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 1, 2)
    return x, y

def cnn_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    cnn = CNN()
    cnn.train(x_train, y_train, 20)

    test_error = 0
    for x, y in zip(x_test, y_test):
        y_pred = cnn.forward(x)
        test_error += cnn.loss.calculate(y_pred, y)
    print(test_error / len(y_pred))