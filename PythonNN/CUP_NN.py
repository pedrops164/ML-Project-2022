from LossFunctions import MEE
from ActivationFunctions import Activation_Linear
from NN import NN

class CUP_NN(NN):
    def __init__(self, pg):
        super().__init__(9, 2, pg, Activation_Linear(), MEE())

