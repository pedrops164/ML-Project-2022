from ActivationFunction1 import Activation_ReLU
from ActivationFunction3 import Activation_Linear
from Layer import Layer_Dense
import numpy as np
from LossMeanSquare import LossMeanSquare
from LossMeanEuclidianError import MEE
from ActivationLoss1 import Activation_Linear_Loss_LMS
import matplotlib.pyplot as plt
from SGD import SGD

class NN1:
    def __init__(self, n_inputs, n_outputs, param_config):
        # number of hidden layers
        self.n_hiddenlayers = param_config.n_hl
        # number of neurons per hidden layer
        self.neurons_per_hidden_layer = param_config.neurons_per_hl
        # number of iterations (epochs)
        self.n_it = param_config.n_it
        # initial learning rate value
        self.lr = param_config.lr
        # learning rate decay value
        self.lr_decay = param_config.lr_decay
        # momentum value
        self.m = param_config.momentum
        # minimum learning rate
        self.min_lr = param_config.min_lr
        self.hidden_layers = []
        if self.n_hiddenlayers == 0:
            # in this case there are no hidden layers, therefore it's just the input layer,
            # and the output layer
            # Layer contains the weights and biases between two layers, therefore one Layer
            # object is enough
            self.first_layer = None
            self.last_layer = Layer_Dense(n_inputs, n_outputs)
        else:
            # in this case there are hidden layers, so we initialize them and we add
            # to the list of hidden_layers 

            # here we initialize the first layer seperately, because the number of inputs
            # is the number of inputs of the neural network
            self.first_layer = Layer_Dense(n_inputs, self.neurons_per_hidden_layer, Activation_ReLU())
            for i in range(self.n_hiddenlayers-1):
                # now we initialize the inner hidden layers, where the number of inputs and
                # outputs are the same (number of neurons per hidden layer)
                hidden_layer = Layer_Dense(self.neurons_per_hidden_layer, self.neurons_per_hidden_layer, Activation_ReLU())
                self.hidden_layers.append(hidden_layer)

            self.last_layer = Layer_Dense(self.neurons_per_hidden_layer, n_outputs, Activation_Linear())

        # we define the loss function
        self.loss = MEE()

    # we propagate forward the inputs, and we return the loss, that's why we receive
    # the real outputs
    def forward(self, inputs, Y):
        if self.first_layer != None:
            self.first_layer.forward(inputs)
            A1 = self.first_layer.output
            inputs = A1

        for layer in self.hidden_layers:
            layer.forward(inputs)
            inputs = layer.output

        self.last_layer.forward(inputs)
        # we calculate loss
        loss = self.loss.calculate(self.last_layer.output, Y)
        return loss

    # we back propagate the gradients, changing the values of the weights and biases of the layers
    def back_prop(self, Y):

        self.loss.backprop(Y)
        self.last_layer.backprop(self.loss.inputs_deriv)
        previous_layer = self.last_layer

        for layer in reversed(self.hidden_layers):
            layer.backprop(previous_layer.inputs_deriv)
            previous_layer = layer

        if self.first_layer != None:
            self.first_layer.backprop(previous_layer.inputs_deriv)


    def adjust_parameters(self, param_adjuster):
        param_adjuster.decay_lr()

        if self.first_layer != None:
            param_adjuster.adjust_parameters(self.first_layer)
        
        for layer in self.hidden_layers:
            param_adjuster.adjust_parameters(layer)

        param_adjuster.adjust_parameters(self.last_layer)
        param_adjuster.increase_iteration()


    def gradient_descent(self, X, Y):

        param_adjuster = SGD(learning_rate=self.lr, decay=self.lr_decay, momentum=self.m, min_lr = self.min_lr)
        # train_size = range(self.n_it)
        # train_Y_data = [] # loss
        # test_Y_data = [] # loss
        for i in range(self.n_it):
            # loss_validation = self.forward(validation_X, validation_Y)
            # test_Y_data.append(np.mean(loss_validation))
            loss_empirical = self.forward(X, Y)
            self.back_prop(Y)
            self.adjust_parameters(param_adjuster)
            # train_Y_data.append(np.mean(loss_empirical))
            #if i % 5 == 0:
            #    print("Iteration: ", i)
            #    print(loss_empirical)

        '''
        plt.plot(train_size, train_Y_data, '--', color="#111111", label="Training loss")
        plt.plot(train_size, test_Y_data, color="#111111", label="Validation loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title("Learning curves")
        plt.axis([0,iterations,0,20])
        plt.show()
        '''

        loss_empirical = self.forward(X, Y)
        # loss_validation = self.forward(validation_X, validation_Y)
        return loss_empirical

    def getParamConfig(self):
        return "Number of hidden layer: " + str(self.n_hiddenlayers) + "\n" + \
               "Number of neurons per hidden layer: " + str(self.neurons_per_hidden_layer) + "\n" + \
               "Number of epochs: " + str(self.n_it) + "\n" + \
               "Initial learning rate: " + str(self.lr) + "\n" + \
               "Learning Rate decay: " + str(self.lr_decay) + "\n" + \
               "Momentum value: " + str(self.m) + "\n" + \
               "Minimum learning rate: " + str(self.min_lr) + "\n"