from ActivationFunction1 import Activation_ReLU
from ActivationFunction3 import Activation_Linear
from Layer import Layer_Dense
import math
import numpy as np
from LossMeanSquare import LossMeanSquare
from LossMeanEuclidianError import MEE
import matplotlib.pyplot as plt
from Adjuster import ParameterAdjuster

class NN1:
    def __init__(self, n_inputs, n_outputs, param_config, activ_out, loss_funct, acc_funct=None):
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
        # lambda for regularization
        self.lambda_param = param_config.lambda_param
        # batch size
        self.batch_size = param_config.batch_size

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

            self.last_layer = Layer_Dense(self.neurons_per_hidden_layer, n_outputs, activ_out)

        # we define the loss function
        # self.loss = MEE()
        self.loss = loss_funct
        self.accuracy = acc_funct

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
        loss += self.regularization_loss()
        return self.last_layer.output, loss # returns the predicted output

    def regularization_loss(self):
        weight_sum = 0
        if self.lambda_param > 0:
            if self.first_layer != None:
                weight_sum += np.sum(self.first_layer.weights * self.first_layer.weights)

            for layer in self.hidden_layers:
                weight_sum += np.sum(layer.weights * layer.weights)

            weight_sum += np.sum(self.last_layer.weights * self.last_layer.weights)

        return self.lambda_param * weight_sum
        

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

        param_adjuster = ParameterAdjuster(learning_rate=self.lr, decay=self.lr_decay, momentum=self.m, min_lr = self.min_lr, lambda_param = self.lambda_param)
        # train_size = range(self.n_it)
        # train_Y_data = [] # loss
        # test_Y_data = [] # loss
        if self.batch_size == 0:
            self.batch_size = len(X)
        n_batches = math.floor(len(X) / self.batch_size)
        for i in range(self.n_it):
            random_permutation = np.random.permutation(len(X))
            X_shuffled = X[random_permutation]
            Y_shuffled = Y[random_permutation]

            for j in range(n_batches):
                batch_start = j * self.batch_size
                batch_end = (j+1) * self.batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                Y_batch = Y_shuffled[batch_start:batch_end]

                # loss_validation = self.forward(validation_X, validation_Y)
                # test_Y_data.append(loss_validation)
                predicted_Y, loss = self.forward(X_batch, Y_batch)
                self.back_prop(Y_batch)
                self.adjust_parameters(param_adjuster)
                # train_Y_data.append(loss_empirical)
                if i % 5 == 0:
                    print("Iteration: ", i, ", Batch: ", j)
                    self.print_measures(predicted_Y, Y_batch)

    # this function trains the neural network on dataset 1, and build a plot graph
    # comparing the learning curves of datasets 1 and 2
    # also receives as input the path to the file to save the plot
    def plot_learning_curves(self, X1, Y1, X2, Y2, measure_function, filepath):
        param_adjuster = ParameterAdjuster(learning_rate=self.lr, decay=self.lr_decay, momentum=self.m, min_lr = self.min_lr, lambda_param = self.lambda_param)
        train_size = range(self.n_it)
        train_Y_data = [] # measure (loss or accuracy)
        test_Y_data = [] # measure (loss or accuracy)
        
        #for each epoch we train
        for i in range(self.n_it):
            # we calculate the measure for the test dataset, not being trained,
            # and we add to the learning curve
            Y2_predicted, loss = self.forward(X2, Y2)
            data2_measure = measure_function.calculate(Y2_predicted, Y2)
            test_Y_data.append(data2_measure)

            # we calculate the measure for the training dataset, and we add
            # to the learning curve
            Y1_predicted, loss = self.forward(X1, Y1)
            data1_measure = measure_function.calculate(Y1_predicted, Y1)
            train_Y_data.append(data1_measure)

            #we backprop and adjust the parameters of the layers
            self.back_prop(Y1)
            self.adjust_parameters(param_adjuster)
            if i % 5 == 0:
                print("Iteration: ", i)
                self.print_measures(Y1_predicted, Y1)
                
        plt.plot(train_size, train_Y_data, '--', color="#111111", label="Training loss")
        plt.plot(train_size, test_Y_data, color="#111111", label="Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel(measure_function.title())
        plt.title("Learning Curves")
        # plt.axis([0,iterations,0,20])
        plt.savefig(filepath)

        '''
        plt.plot(train_size, train_Y_data, '--', color="#111111", label="Training loss")
        plt.plot(train_size, test_Y_data, color="#111111", label="Validation loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title("Learning curves")
        plt.axis([0,iterations,0,20])
        plt.show()
        '''

        output, loss_empirical = self.forward(X1, Y1)
        # loss_validation = self.forward(validation_X, validation_Y)
        return loss_empirical

    def print_measures(self, predicted_Y, target_Y):
        print("Loss: " + str(self.loss.calculate(predicted_Y, target_Y)))
        if (self.accuracy != None):
            print("Accuracy: " + str(self.accuracy.calculate(predicted_Y, target_Y)))

    def getParamConfig(self):
        return "Number of hidden layer: " + str(self.n_hiddenlayers) + "\n" + \
               "Number of neurons per hidden layer: " + str(self.neurons_per_hidden_layer) + "\n" + \
               "Number of epochs: " + str(self.n_it) + "\n" + \
               "Initial learning rate: " + str(self.lr) + "\n" + \
               "Learning Rate decay: " + str(self.lr_decay) + "\n" + \
               "Momentum value: " + str(self.m) + "\n" + \
               "Minimum learning rate: " + str(self.min_lr) + "\n"