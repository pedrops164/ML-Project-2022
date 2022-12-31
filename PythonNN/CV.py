import numpy as np
from NN1 import NN1
from ParamConfig import ParamConfig
import math

'''
Receives ParamConfig object, the input data, and an int k (k-cross validation)
Returns the mean training error, and validation error
'''
def cross_validation(pg, X, Y, k):
    #it is assumed that the input data is shuffled
    block_size = math.floor(len(X) / k)
    n_inputs = 9
    n_outputs = 2

    training_errors = []
    validation_errors = []

    for i in range(k):
        nn = NN1(n_inputs, n_outputs, pg.n_hl, pg.neurons_per_hl)
        val_start = i*block_size
        val_end = (i+1)*block_size
        validation_X = X[val_start:val_end]
        validation_Y = Y[val_start:val_end]
        training_X = np.concatenate((X[:val_start], X[val_end:]))
        training_Y = np.concatenate((Y[:val_start], Y[val_end:]))

        training_error, validation_error = nn.gradient_descent(training_X, training_Y,
                                                               validation_X, validation_Y,
                                                               param_config=pg)
        training_errors.append(training_error)
        validation_errors.append(validation_error)

    print(training_errors)
    print(validation_errors)
    return np.mean(training_errors), np.mean(validation_errors)