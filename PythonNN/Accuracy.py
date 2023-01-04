import numpy as np

class Accuracy:
    def calculate(self, output, y):
        # self.output = output
        # here we assume the threshold is at 0. for example output of tanh activation function

        # we create matrix with same shape as output, filled with zeros
        accuracy = np.zeros_like(output)
        # we set to 1 the positions where the output is higher than 0 (0 is threshold)
        accuracy[output > 0] = 1
        return np.mean(accuracy==y) # value between 0 and 1. closer to 1 is more accurate

