import numpy as np

class Loss:
    def calculate(self, output, y):
        self.output = output
        losses = self.forward(output, y) # returns 1d array
        return np.mean(losses)

    def title(self):
        return "Loss"