from ActivationFunction1 import Activation_ReLU
from Layer import Layer_Dense
import numpy as np
from LossMeanSquare import LossMeanSquare
from ActivationLoss1 import Activation_Linear_Loss_LMS
import matplotlib.pyplot as plt
from SGD import SGD

class NN1:
    def __init__(self, n_inputs, n_outputs, n_hiddenlayers, neurons_per_hidden_layer):
        self.hidden_layers = []
        if n_hiddenlayers == 0:
            self.first_layer = None
            self.last_layer = Layer_Dense(n_inputs, n_outputs)
        else:
            self.first_layer = Layer_Dense(n_inputs, neurons_per_hidden_layer, Activation_ReLU())
            for i in range(n_hiddenlayers-1):
                hidden_layer = Layer_Dense(neurons_per_hidden_layer, neurons_per_hidden_layer, Activation_ReLU())
                self.hidden_layers.append(hidden_layer)

            self.last_layer = Layer_Dense(neurons_per_hidden_layer, n_outputs)

        # self.layer1 = Layer_Dense(n_inputs, 5, Activation_ReLU())
        # self.layer2 = Layer_Dense(5, n_outputs)
        self.activ_loss = Activation_Linear_Loss_LMS()


    def forward(self, inputs, Y):
        if self.first_layer != None:
            self.first_layer.forward(inputs)
            A1 = self.first_layer.output
            inputs = A1

        for layer in self.hidden_layers:
            layer.forward(inputs)
            inputs = layer.output

        self.last_layer.forward(inputs)
        Z_last = self.last_layer.output
        loss = self.activ_loss.forward(Z_last, Y)
        return loss

        # self.layer1.forward(inputs)
        # A1 = self.layer1.output
        # self.layer2.forward(A1)
        # Z2 = self.layer2.output
        # loss = self.activ_loss.forward(Z2, Y)
        # return loss
    def back_prop(self, Y):
        # n_samples = Y.shape[0] # number of examples

        self.activ_loss.backprop(Y)
        # self.layer2.backprop(self.activ_loss.inputs_deriv)
        self.last_layer.backprop(self.activ_loss.inputs_deriv)
        previous_layer = self.last_layer

        for layer in reversed(self.hidden_layers):
            layer.backprop(previous_layer.inputs_deriv)
            previous_layer = layer

        if self.first_layer != None:
            self.first_layer.backprop(previous_layer.inputs_deriv)

        # self.layer1.backprop(self.layer2.inputs_deriv)

    def adjust_parameters(self, param_adjuster):
        param_adjuster.decay_lr()

        if self.first_layer != None:
            # self.first_layer.update_params(alpha)
            param_adjuster.adjust_parameters(self.first_layer)
        
        for layer in self.hidden_layers:
            # layer.update_params(alpha)
            param_adjuster.adjust_parameters(layer)

        # self.last_layer.update_params(alpha)
        param_adjuster.adjust_parameters(self.last_layer)
        param_adjuster.increase_iteration()


    def gradient_descent(self, X, Y, test_X, test_Y, iterations, lr, lr_decay, m):
        param_adjuster = SGD(learning_rate=lr, decay=lr_decay, momentum=m)
        train_size = range(iterations)
        train_Y_data = [] # loss
        test_Y_data = [] # loss
        for i in range(iterations):
            loss_validation = self.forward(test_X, test_Y)
            test_Y_data.append(np.mean(loss_validation))
            loss_empirical = self.forward(X, Y)
            self.back_prop(Y)
            self.adjust_parameters(param_adjuster)
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
