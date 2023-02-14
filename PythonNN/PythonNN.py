import numpy as np
import time

from Model import Model
from CV import CrossValidation
from Grid import Grid
from MONK_NN import *
from CUP_NN import CUP_NN
from CNN import cnn_mnist

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
	output_file_name = "outputs/ErasmusNeuralNetwork_ML-CUP22-TS.csv"
	output_file = open(output_file_name, "w")
	
	# take out the comment lines
	csvreader = csv.reader(blind_set)
	header = []
	header = next(csvreader)
	for i in range(6):
		next(csvreader)

	output_file.write("# Pedro Sousa, Diana Mateus, Nikolai Hoffmann\n")
	output_file.write("# ErasmusNeuralNetwork\n")
	output_file.write("# ML-CUP22\n")
	output_file.write("# 05/01/2023\n")

	for row in csvreader:
		id = row[0]
		input = row[1:]
		input = np.array(input, dtype=float)
		output = model.calculate_output(input)
		line = str(id) + "," + str(output[0]) + "," + str(output[1])
		output_file.write(line)
		output_file.write("\n")

	output_file.close()

def ml_cup():
	start = time.time()

	train_X, train_Y, test_X, test_Y = initialize_cup_tr('inputs/ML-CUP22-TR.csv', 0.25)
	
	config_list = []
	
	grid = Grid([2,4], # number of hidden layers
				[25,50,75], # neurons per hidden layer
				[1500,2000,2500,3000,3500], # number of iterations
				[0.005], # initial learning rate
				[0], # learning rate decay
				[0.7, 0.8], # momentum value
				[0], # minimum learning rate
				[0,0.00005,0.0001], # l2 regularization lambda value
				[0]) # batch size
	
	logfile = open("outputs/log.txt", "w")
	
	model = Model(logfile)
	model.model_selection(train_X, train_Y, test_X, test_Y, grid.configs, CrossValidation(k=4, runs=1), top_n=9)
	model.model_assessment(test_X, test_Y)
	model.reset_params()
	
	#this time we train the best models with the whole dataset
	X, Y, empty_X, empty_Y = initialize_cup_tr('inputs/ML-CUP22-TR.csv', 0)
	model.retrain(X, Y)
	
	model.print_model()
	
	finalize_cup_file('inputs/ML-CUP22-TS.csv', model)
	
	end = time.time()
	
	logfile.write("elapsed time: " +  str(end - start))
	logfile.close()


# ml_cup()
cnn_mnist()