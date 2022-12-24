import numpy as np

class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)