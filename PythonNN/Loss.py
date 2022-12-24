import numpy as np

class Loss:
    def calculate(self, output, y):
        losses = self.forward(output, y)
        return [np.mean(losses[0]), np.mean(losses[1])]