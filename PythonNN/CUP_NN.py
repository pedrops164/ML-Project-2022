from LossMeanEuclidianError import MEE
from ActivationFunction3 import Activation_Linear
from NN1 import NN1

class CUP_NN(NN1):
    def __init__(self, pg):
        super().__init__(9, 2, pg, Activation_Linear(), MEE())

