from Layer import ConvolutionalLayer, ReshapeLayer, Dense, MaxPooling
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
            ConvolutionalLayer((5,26,26), 3, 5),
            Sigmoid(),
            ReshapeLayer((5,24,24), (1, 5*24*24)),
            Dense(5*24*24, 100, Sigmoid()),
            Dense(100,10, Sigmoid())            
        ]
        # self.layers = [
        #     ConvolutionalLayer((1,28,28), 3, 5),
        #     Sigmoid(),
        #     MaxPooling((5,26,26), ksize=(2,2)),
        #     ConvolutionalLayer((5,13,13), 3, 5),
        #     Sigmoid(),
        #     ReshapeLayer((5,11,11), (1, 5*11*11)),
        #     Dense(5*11*11, 100, Sigmoid()),
        #     Dense(100,10, Sigmoid())            
        # ]

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


def preprocess_mnist(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 1, 10)
    return x[:limit], y[:limit]

def cnn_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist(x_train, y_train, 1000)
    x_test, y_test = preprocess_mnist(x_test, y_test, 1000)

    cnn = CNN()
    cnn.train(x_train, y_train, 20)

    test_error = 0
    accuracy = 0
    for x, y in zip(x_test, y_test):
        y_pred = cnn.forward(x)
        prediction = np.argmax(y_pred)
        true_prediction = np.argmax(y)
        if prediction == true_prediction:
            accuracy += 1
        test_error += cnn.loss.calculate(y_pred, y)
    print(test_error / len(y_pred))
    print("accuracy: ", accuracy / (len(x_test)))