import numpy as np
from NN1 import NN1
from ParamConfig import ParamConfig
import math

class CrossValidation:
    def __init__(self, k):
        # k fold cross validation
        self.k = k

    '''
    Receives ParamConfig object, the input data, and an int k (k-cross validation)
    Returns the mean training error, and validation error
    '''
    def cross_validation(self, pg, X, Y):
        #it is assumed that the input data is shuffled
        block_size = math.floor(len(X) / self.k)
        print(X.shape)
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]
    
        training_errors = []
        validation_errors = []
    
        for i in range(self.k):
            nn = NN1(n_inputs, n_outputs, pg)
            val_start = i*block_size
            val_end = (i+1)*block_size
            validation_X = X[val_start:val_end]
            validation_Y = Y[val_start:val_end]
            training_X = np.concatenate((X[:val_start], X[val_end:]))
            training_Y = np.concatenate((Y[:val_start], Y[val_end:]))
    
            training_error = nn.gradient_descent(training_X, training_Y)
            validation_error = nn.forward(validation_X, validation_Y)
            training_errors.append(training_error)
            validation_errors.append(validation_error)
    
        neural_network = NN1(n_inputs, n_outputs, pg)
        #we train on all training+validation data this time
        neural_network.gradient_descent(X, Y)

        return neural_network, np.mean(training_errors), np.mean(validation_errors)
    
    
    # def cross_validation(pg, X, Y, k):