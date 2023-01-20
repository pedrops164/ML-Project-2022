from matplotlib import pyplot

from ActivationFunctions import Activation_ReLU, Activation_Linear, Sigmoid
from Layer import Layer
import math
import numpy as np
from LossFunctions import MEE, MSE
import matplotlib.pyplot as plt
from PlotMaker import make_plot
from Adjuster import ParameterAdjuster
from Accuracy import Accuracy


class NN:
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
        # TR learning curve array for loss
        self.tr_loss_lc = []
        # TS learning curve array for loss
        self.ts_loss_lc = []

        self.param_adjuster = ParameterAdjuster(
            learning_rate=param_config.lr,
           decay=param_config.lr_decay,
          momentum=param_config.momentum,
         min_lr = param_config.min_lr,
        lambda_param = param_config.lambda_param)

        self.hidden_layers = []
        if self.n_hiddenlayers == 0:
            # in this case there are no hidden layers, therefore it's just the input layer,
            # and the output layer
            # Layer contains the weights and biases between two layers, therefore one Layer
            # object is enough
            self.first_layer = None
            self.last_layer = Layer(n_inputs, n_outputs)
        else:
            # in this case there are hidden layers, so we initialize them and we add
            # to the list of hidden_layers 

            # here we initialize the first layer seperately, because the number of inputs
            # is the number of inputs of the neural network
            self.first_layer = Layer(n_inputs, self.neurons_per_hidden_layer, Sigmoid())
            for i in range(self.n_hiddenlayers-1):
                # now we initialize the inner hidden layers, where the number of inputs and
                # outputs are the same (number of neurons per hidden layer)
                hidden_layer = Layer(self.neurons_per_hidden_layer, self.neurons_per_hidden_layer, Sigmoid())
                self.hidden_layers.append(hidden_layer)

            self.last_layer = Layer(self.neurons_per_hidden_layer, n_outputs, activ_out)

        # we define the loss function
        # self.loss = MEE()
        self.loss = loss_funct
        self.accuracy = acc_funct

    # we propagate forward the inputs, and we return the loss, that's why we receive
    # the real outputs
    def forward(self, inputs, Y):
        # we calculate loss
        expected_output = self.output(inputs)
        loss = self.loss.calculate(expected_output, Y)
        loss += self.regularization_loss()
        return expected_output, loss # returns the predicted output

    def output(self, inputs):
        if self.first_layer != None:
            self.first_layer.forward(inputs)
            A1 = self.first_layer.output
            inputs = A1

        for layer in self.hidden_layers:
            layer.forward(inputs)
            inputs = layer.output

        self.last_layer.forward(inputs)
        return self.last_layer.output

    def reset_params(self):
        if self.first_layer != None:
            self.first_layer.reset_params()

        for layer in self.hidden_layers:
            layer.reset_params()

        self.last_layer.reset_params()

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
        # self.last_layer.backprop(np.mean(self.loss.inputs_deriv, axis=0))
        self.last_layer.backprop(self.loss.inputs_deriv)
        previous_layer = self.last_layer

        for layer in reversed(self.hidden_layers):
            layer.backprop(previous_layer.inputs_deriv)
            previous_layer = layer

        if self.first_layer != None:
            self.first_layer.backprop(previous_layer.inputs_deriv)


    def adjust_parameters(self, batch_multiplier):
        self.param_adjuster.decay_lr()

        if self.first_layer != None:
            self.param_adjuster.adjust_parameters(self.first_layer, batch_multiplier)
        
        for layer in self.hidden_layers:
            self.param_adjuster.adjust_parameters(layer, batch_multiplier)

        self.param_adjuster.adjust_parameters(self.last_layer, batch_multiplier)
        self.param_adjuster.increase_iteration()

    def train(self, X, Y, print_progress=False):

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

                predicted_Y, loss = self.forward(X_batch, Y_batch)
                self.back_prop(Y_batch)
                self.adjust_parameters(1/n_batches)
                if i%5==0 and print_progress:
                    print("Iteration: ", i, ", Batch: ", j)
                    self.print_measures(predicted_Y, Y_batch)

        # output, loss_empirical = self.forward(X1, Y1)
        # # loss_validation = self.forward(validation_X, validation_Y)
        # return loss_empirical

    # this function trains the neural network on dataset 1, and build a plot graph
    # comparing the learning curves of datasets 1 and 2
    # also receives as input the path to the file to save the plot
    def plot_learning_curves(self, X1, Y1, X2, Y2, filepath=None, trials=1, print_progress=False):
        train_size = range(self.n_it)
        train_Y_accuracy = []  # measure (accuracy)
        test_Y_accuracy = []  # measure (accuracy)
        train_Y_loss = []  # measure (loss)
        test_Y_loss = []  # measure (loss)

        if self.batch_size == 0:
            self.batch_size = len(X1)
        n_batches = math.floor(len(X1) / self.batch_size)
        
        #for each epoch we train
        for i in range(self.n_it):
            random_permutation = np.random.permutation(len(X1))
            X1_shuffled = X1[random_permutation]
            Y1_shuffled = Y1[random_permutation]

            for batch in range(n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch+1) * self.batch_size
                X_batch = X1_shuffled[batch_start:batch_end]
                Y_batch = Y1_shuffled[batch_start:batch_end]

                # we calculate the measure for the training dataset, and we add
                # to the learning curve
                Y_pred, loss = self.forward(X_batch, Y_batch)
                #we backprop and adjust the parameters of the layers
                self.back_prop(Y_batch)
                self.adjust_parameters(1/n_batches)

            # here we calculate the loss with the full batch, for both data sets

            # we calculate the measure for the test dataset, not being trained,
            # and we add to the learning curve
            Y2_predicted, loss = self.forward(X2, Y2)
            # data2_measure_accuracy = measure_function.calculate(Y2_predicted, Y2)
            data2_measure_accuracy = Accuracy().calculate(Y2_predicted, Y2)
            data2_measure_loss = MEE().calculate(Y2_predicted, Y2)
            test_Y_accuracy.append(data2_measure_accuracy)
            self.ts_loss_lc.append(data2_measure_loss)
                
            # we calculate the measure for the training dataset, and we add
            # to the learning curve
            Y1_predicted, loss = self.forward(X1, Y1)
            # data1_measure_accuracy = measure_function.calculate(Y1_predicted, Y1)
            data1_measure_accuracy = Accuracy().calculate(Y1_predicted, Y1)
            data1_measure_loss = MEE().calculate(Y1_predicted, Y1)
            train_Y_accuracy.append(data1_measure_accuracy)
            self.tr_loss_lc.append(data1_measure_loss)
            if i % 50 == 0 and print_progress:
                print("Iteration: ", i)
                self.print_measures(Y1_predicted, Y1)
                self.print_measures(Y2_predicted, Y2)


        # now that the nn is trained, we calculate the final measured value
        measured_accuracy_train = []
        measured_accuracy_test = []
        measured_loss_train = []
        measured_loss_test = []
        for t in range(trials):
            Y2_predicted, loss = self.forward(X2, Y2)
            # data2_measure_accuracy = measure_function.calculate(Y2_predicted, Y2)
            data2_measure_accuracy = Accuracy().calculate(Y2_predicted, Y2)
            data2_measure_loss = MEE().calculate(Y2_predicted, Y2)
            measured_accuracy_test.append(data2_measure_accuracy)
            measured_loss_test.append(data2_measure_loss)

            Y1_predicted, loss = self.forward(X1, Y1)
            # data1_measure_accuracy = measure_function.calculate(Y1_predicted, Y1)
            data1_measure_accuracy = Accuracy().calculate(Y1_predicted, Y1)
            data1_measure_loss = MEE().calculate(Y1_predicted, Y1)
            measured_accuracy_train.append(data1_measure_accuracy)
            measured_loss_train.append(data1_measure_loss)


        # make_plot(train_size, train_Y_accuracy, test_Y_accuracy, "Accuracy", "Epochs vs Accuracy", different_file_path)
        if filepath != None:
            make_plot(train_size, self.tr_loss_lc, self.ts_loss_lc, "MEE", "Epochs vs MEE", filepath, ylim=(0,3))

        final_test_measured = [np.mean(measured_accuracy_test), np.mean(measured_loss_test)]
        final_train_measured = [np.mean(measured_accuracy_train), np.mean(measured_loss_train)]

        return final_train_measured, final_test_measured

    # this function trains the neural network on dataset 1, and build a plot graph
    # comparing the learning curves of datasets 1 and 2
    # also receives as input the path to the file to save the plot
    def plot_learning_curvesMonk(self, X1, Y1, X2, Y2, filepath, trials):
        train_size = range(self.n_it)
        train_Y_accuracy = []  # measure (accuracy)
        test_Y_accuracy = []  # measure (accuracy)
        train_Y_loss = []  # measure (loss)
        test_Y_loss = []  # measure (loss)

        if self.batch_size == 0:
            self.batch_size = len(X1)
        n_batches = math.floor(len(X1) / self.batch_size)
        
        #for each epoch we train
        for i in range(self.n_it):
            random_permutation = np.random.permutation(len(X1))
            X1_shuffled = X1[random_permutation]
            Y1_shuffled = Y1[random_permutation]

            for batch in range(n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch+1) * self.batch_size
                X_batch = X1_shuffled[batch_start:batch_end]
                Y_batch = Y1_shuffled[batch_start:batch_end]

                # we calculate the measure for the training dataset, and we add
                # to the learning curve
                Y_pred, loss = self.forward(X_batch, Y_batch)
                #we backprop and adjust the parameters of the layers
                self.back_prop(Y_batch)
                self.adjust_parameters()

            # here we calculate the loss with the full batch, for both data sets

            # we calculate the measure for the test dataset, not being trained,
            # and we add to the learning curve
            Y2_predicted, loss = self.forward(X2, Y2)
            # data2_measure_accuracy = measure_function.calculate(Y2_predicted, Y2)
            data2_measure_accuracy = Accuracy().calculate(Y2_predicted, Y2)
            data2_measure_loss = MSE().calculate(Y2_predicted, Y2)
            test_Y_accuracy.append(data2_measure_accuracy)
            test_Y_loss.append(data2_measure_loss)
                
            # we calculate the measure for the training dataset, and we add
            # to the learning curve
            Y1_predicted, loss = self.forward(X1, Y1)
            # data1_measure_accuracy = measure_function.calculate(Y1_predicted, Y1)
            data1_measure_accuracy = Accuracy().calculate(Y1_predicted, Y1)
            data1_measure_loss = MSE().calculate(Y1_predicted, Y1)
            train_Y_accuracy.append(data1_measure_accuracy)
            train_Y_loss.append(data1_measure_loss)
            if i % 50 == 0:
                print("Iteration: ", i)
                self.print_measures(Y1_predicted, Y1)
                self.print_measures(Y2_predicted, Y2)


        # now that the nn is trained, we calculate the final measured value
        measured_accuracy_train = []
        measured_accuracy_test = []
        measured_loss_train = []
        measured_loss_test = []
        for t in range(trials):
            Y2_predicted, loss = self.forward(X2, Y2)
            # data2_measure_accuracy = measure_function.calculate(Y2_predicted, Y2)
            data2_measure_accuracy = Accuracy().calculate(Y2_predicted, Y2)
            data2_measure_loss = MSE().calculate(Y2_predicted, Y2)
            measured_accuracy_test.append(data2_measure_accuracy)
            measured_loss_test.append(data2_measure_loss)

            Y1_predicted, loss = self.forward(X1, Y1)
            # data1_measure_accuracy = measure_function.calculate(Y1_predicted, Y1)
            data1_measure_accuracy = Accuracy().calculate(Y1_predicted, Y1)
            data1_measure_loss = MSE().calculate(Y1_predicted, Y1)
            measured_accuracy_train.append(data1_measure_accuracy)
            measured_loss_train.append(data1_measure_loss)


        # Plotting the data and saving it
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(train_size, train_Y_accuracy, '--', color="b", label="Training")
        plt.plot(train_size, test_Y_accuracy, color="r", label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel(Accuracy().title())
        plt.title("Epochs vs " + str(Accuracy().title()))
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_size, train_Y_loss, '--', color="b", label="Training")
        plt.plot(train_size, test_Y_loss, color="r", label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel(MSE().title())
        plt.title("Epochs vs " + str(MSE().title()))
        plt.legend()

        """
        plt.subplot(1, 3, 3)
        plt.plot(train_Y_accuracy, test_Y_accuracy, color="g", label="Accuracy (Train vs Test)")
        plt.xlabel('Train Accuracy')
        plt.ylabel('Test Accuracy')
        plt.title("Train vs Test Accuracy")
        plt.legend()
        """

        plt.savefig(filepath)
        plt.show()

        final_test_measured = [np.mean(measured_accuracy_test), np.mean(measured_loss_test)]
        final_train_measured = [np.mean(measured_accuracy_train), np.mean(measured_loss_train)]

        return final_train_measured, final_test_measured

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
               "Minimum learning rate: " + str(self.min_lr) + "\n" + \
               "Lambda: " + str(self.lambda_param) + "\n" + \
               "Batch size: " + str(self.batch_size) + "\n"