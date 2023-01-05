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
from CUP_NN import CUP_NN

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

# This function receives the path to the blind test set of the cup,
# and a model which will calculate the target value for each output,
# and produces the final file with our target outputs
def finalize_cup_file(blind_set_path, model):
	blind_set = open(blind_set_path)
	output_file_name = "outputs/team-name_ML-CUP22-TS.csv"
	output_file = open(output_file_name, "a")
	
	# take out the comment lines
	csvreader = csv.reader(blind_set)
	header = []
	header = next(csvreader)
	for i in range(6):
		next(csvreader)

	output_file.write("Pedro Sousa, Diana Mateus, Nikolai Hoffmann")
	output_file.write("Team name")
	output_file.write("ML-CUP22")
	output_file.write("05/01/2023")

	for row in csvreader:
		id = row[0]
		input = row[1:]
		output = model.calculate_output(input)
		line = str(id) + "," + str(output[0]) + "," + str(output[1])
		output_file.write(line)

	output_file.close()


"""
# USE THIS CODE TO TRY OUT DIFFERENT CONFIGS FOR THE MONK
X1, Y1 = parse_monk("inputs/monks-1.train")
X2, Y2 = parse_monk("inputs/monks-1.test")

n_hl = 1  # number of hidden layers
neurons_per_hl = 15  # neurons per hidden layer
n_it = 500  # number of iterations
lr = 0.76  # initial learning rate
lr_decay = 0  # learning rate decay
momentum = 0.83  # momentum value
min_lr = 0  # minimum learning rate
lambda_param = 0  # l2 regularization lambda value
batch_size = 0  # batch size
pg = ParamConfig(n_hl, neurons_per_hl, n_it, lr, lr_decay, momentum, min_lr, lambda_param, batch_size)

nn = MONK_NN(pg)
final_measure_train, final_measure_test = nn.plot_learning_curves(X1, Y1, X2, Y2, Accuracy(), "plots/monk1_accuracy.png", 10)

print(final_measure_train)
print(final_measure_test)

"""
# USE THIS CODE TO TRY OUT DIFFERENT CONFIGS FOR THE CUP
n_hl = 1  # number of hidden layers
neurons_per_hl = 16  # neurons per hidden layer
n_it = 1501  # number of iterations
lr = 0.5  # initial learning rate
lr_decay = 0  # learning rate decay
momentum = 0.7  # momentum value
min_lr = 0  # minimum learning rate
lambda_param = 0  # l2 regularization lambda value
batch_size = 0  # batch size
pg = ParamConfig(n_hl, neurons_per_hl, n_it, lr, lr_decay, momentum, min_lr, lambda_param, batch_size)
#initializes the training/validation and testing set
train_X, train_Y, test_X, test_Y = initialize_cup_tr('inputs/ML-CUP22-TR.csv', 0.2)
cv = CrossValidation(k=4, runs=3)
# does k fold cross validation with given runs, with the training/validation set, and
# gives the final training and validation errors
nn, tr_errors, vl_errors = cv.cross_validation(CUP_NN, pg, train_X, train_Y)
print(tr_errors) # final, average training errors
print(vl_errors) # final, average validation errors

# IF YOU WANT TO BUILD THE PLOT FOR A CERTAIN ParamConfig pg, you do
# nn = CUP_NN(pg)
# final_measure_train, final_measure_test = nn.plot_learning_curves(X1, Y1, X2, Y2, MEE(), "plots/cup_mee.png", 5)
# 
# print(final_measure_train)
# print(final_measure_test)

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