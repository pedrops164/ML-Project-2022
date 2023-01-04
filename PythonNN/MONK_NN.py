from NN1 import NN1
from ActivationFunction4 import Tanh
from LossBinaryCrossEntropy import BCE
from Accuracy import Accuracy
import csv
from sklearn.preprocessing import OneHotEncoder
from ParamConfig import ParamConfig
import numpy as np

class MONK_NN(NN1):
    def __init__(self, pg):
        super().__init__(17, 1, pg, Tanh(), BCE(), acc_funct=Accuracy())


def parse_monk(filepath):
    
    file = open(filepath)
    csvreader = csv.reader(file, delimiter=" ")

    """
    7. Attribute information:
    1. class: 0, 1 
    2. a1:    1, 2, 3
    3. a2:    1, 2, 3
    4. a3:    1, 2
    5. a4:    1, 2, 3
    6. a5:    1, 2, 3, 4
    7. a6:    1, 2
    8. Id:    (A unique symbol for each instance)
    """
    X_data = []
    Y_data = []
    for row in csvreader:
        label = int(row[1])
        input = []
        input.append(int(row[2]))
        input.append(int(row[3]))
        input.append(int(row[4]))
        input.append(int(row[5]))
        input.append(int(row[6]))
        input.append(int(row[7]))

        X_data.append(input)
        Y_data.append([label])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    X_data = OneHotEncoder().fit_transform(X_data).toarray().astype(np.float32)


    return X_data, Y_data