import numpy as np
import csv

from Model import Model
from Layer import Layer_Dense
from ActivationFunction1 import Activation_ReLU
from ActivationFunction2 import Activation_Softmax
from MLP import MLP
from NN1 import NN1
from CV import CrossValidation
from Grid import Grid
from ParamConfig import ParamConfig

file = open('inputs/ML-CUP22-TR.csv')

csvreader = csv.reader(file)
header = []
header = next(csvreader)
for i in range(6):
	next(csvreader)

# at this point, the useless lines have been skipped

total_data = []

train_X = []
train_Y = []
test_X = []
test_Y = []
for row in csvreader:
	total_data.append(row[1:]) # skip the id, and add to the container with all data

np.random.shuffle(total_data)

# this variable holds the proportion of data that will be used for testing
# the final model
test_size_proportion = 0.2 # value between 0 and 1
tr_vl_size = len(total_data) * (1-test_size_proportion)

for index, row in enumerate(total_data):
	if index < tr_vl_size:
		train_X.append(row[:9])
		train_Y.append(row[9:])
	else:
		test_X.append(row[:9])
		test_Y.append(row[9:])

# here we just convert all arrays to numpy arrays
train_X = np.array(train_X, dtype=float)
train_Y = np.array(train_Y, dtype=float)
test_X = np.array(test_X, dtype=float)
test_Y = np.array(test_Y, dtype=float)

grid = Grid([2], # number of hidden layers
			[64], # neurons per hidden layer
			[2001], # number of iterations
			[0.8], # initial learning rate
			[0], # learning rate decay
			[0.7], # momentum value
			[0]) # minimum learning rate

model = Model()
model.model_selection(train_X, train_Y, grid, CrossValidation(k=4, runs=5))
model.model_assessment(test_X, test_Y)
model.print_model()