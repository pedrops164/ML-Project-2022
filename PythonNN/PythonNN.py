import numpy as np

from Model import Model
from Layer import Layer_Dense
from ActivationFunction1 import Activation_ReLU
from LossBinaryCrossEntropy import BCE
from Accuracy import Accuracy
from NN1 import NN1
from CV import CrossValidation
from Grid import Grid
from ParamConfig import ParamConfig
from MONK_NN import *

def initialize_cup_tr(path, test_size_proportion):

	file = open(path)
	
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
	
	# this variable holds the proportion of data that will be used for training
	# and validating the final model
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

	return train_X, train_Y, test_X, test_Y


X1, Y1 = parse_monk("inputs/monks-1.train")
X2, Y2 = parse_monk("inputs/monks-1.test")

pg = ParamConfig(1,10,500,0.5,0,0.9,0,0,0)
# cv = CrossValidation(k=4, runs=1)
# nn, tr_errors, vl_errors = cv.cross_validation(pg,X1,Y1)
# output, final_loss = nn.forward(X1, Y1)
# final_accuracy = nn.accuracy.calculate(output, Y1)
# print("final loss: " + str(final_loss))
# print("final accuracy: " + str(final_accuracy))

nn = MONK_NN(pg)
#nn.gradient_descent(X1, Y1) # trains Neural Network
#output_training, training_error = nn.forward(X1, Y1)
#output_validation, validation_error = nn.forward(X2, Y2)
nn.plot_learning_curves(X1, Y1, X2, Y2, Accuracy(), "plots/monk1_accuracy.png")


"""

train_X, train_Y, test_X, test_Y = initialize_cup_tr('inputs/ML-CUP22-TR.csv', 0.2)

grid = Grid([1], # number of hidden layers
			[16], # neurons per hidden layer
			[1501], # number of iterations
			[0.5], # initial learning rate
			[0], # learning rate decay
			[0.7], # momentum value
			[0], # minimum learning rate
			[0], # l2 regularization lambda value
			[0]) # batch size

model = Model()
model.model_selection(train_X, train_Y, grid, CrossValidation(k=4, runs=1))
model.model_assessment(test_X, test_Y)
model.print_model()

"""


