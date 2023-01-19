from NN import NN
from LossFunctions import BCE
from Accuracy import Accuracy
import csv
from ActivationFunctions import Sigmoid
from ParamConfig import ParamConfig
import numpy as np

class MONK_NN(NN):
    def __init__(self, pg):
        super().__init__(17, 1, pg, Sigmoid(), BCE(), acc_funct=Accuracy())


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
        input = getClass1ofk(1, int(row[2]))
        input = np.concatenate((input, getClass1ofk(2, int(row[3]))))
        input = np.concatenate((input, getClass1ofk(3, int(row[4]))))
        input = np.concatenate((input, getClass1ofk(4, int(row[5]))))
        input = np.concatenate((input, getClass1ofk(5, int(row[6]))))
        input = np.concatenate((input, getClass1ofk(6, int(row[7]))))
        X_data.append(input)
        Y_data.append([label])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    return X_data, Y_data

def getClass1ofk(class_id, value):
    if class_id==1 or class_id==2 or class_id==4:
        size = 3
    elif class_id==3 or class_id==6:
        size = 2
    else:
        size = 4
    arr = np.zeros(size)
    arr[value-1] = 1
    return arr
