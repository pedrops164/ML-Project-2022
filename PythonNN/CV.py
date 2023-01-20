import numpy as np
from NN import NN
from CUP_NN import CUP_NN
from MONK_NN import MONK_NN
from ParamConfig import ParamConfig
import math

class CrossValidation:
    def __init__(self, k, runs=1):
        # k fold cross validation
        self.k = k
        self.runs = runs

    '''
    Receives ParamConfig object, the input data, and an int k (k-cross validation)
    Returns the mean training error, and validation error
    '''
    def cross_validation(self, nn_class, pg, X, Y, test_X=None, test_Y=None, plot_file_path=None):
        #it is assumed that the input data is shuffled
        block_size = math.floor(len(X) / self.k)
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]
    
        training_errors = []
        validation_errors = []
    
        for i in range(self.k):
            val_start = i*block_size
            val_end = (i+1)*block_size
            validation_X = X[val_start:val_end]
            validation_Y = Y[val_start:val_end]
            training_X = np.concatenate((X[:val_start], X[val_end:]))
            training_Y = np.concatenate((Y[:val_start], Y[val_end:]))
            for j in range(self.runs):
                '''
                For each cross validation,
                we are doing a given amount of runs, where we initialize the neural
                network each time. This makes it so that the weights and biases initialized
                each time are different (randomized), and therefore there's less chance of
                us getting penalised because of a lucky or unlucky
                initialization of weights and biases.
                '''
                nn = nn_class(pg)
    
                nn.train(training_X, training_Y, print_progress=False) # trains Neural Network
                output_training, training_error = nn.forward(training_X, training_Y)
                output_validation, validation_error = nn.forward(validation_X, validation_Y)
                training_errors.append(training_error)
                validation_errors.append(validation_error)
    
        neural_network = nn_class(pg)
        #we train on all training+validation data this time
        if plot_file_path != None:
            neural_network.plot_learning_curves(X, Y, test_X, test_Y,plot_file_path,trials=1,print_progress=True)

        return neural_network, np.mean(training_errors), np.mean(validation_errors)