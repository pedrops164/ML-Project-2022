import numpy as np
from NN1 import NN1
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
    def cross_validation(self, pg, X, Y, table):
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
                nn = MONK_NN(pg)
    
                nn.gradient_descent(training_X, training_Y, table) # trains Neural Network
                output_training, training_error = nn.forward(training_X, training_Y)
                output_validation, validation_error = nn.forward(validation_X, validation_Y)
                training_errors.append(training_error)
                validation_errors.append(validation_error)
    
        neural_network = MONK_NN(pg)
        #we train on all training+validation data this time
        neural_network.gradient_descent(X, Y, table)

        return neural_network, np.mean(training_errors), np.mean(validation_errors)
    
    
    # def cross_validation(pg, X, Y, k):