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

# np.random.seed(0)
'''
inputs = [1, 2, 3, 2.5]

inputs1 = [[1, 2, 3, 2.5],
		   [2, 5, -1, 2],
		   [-1.5, 2.7, 3.3, -0.8]]


weights1 = [0.5, 0.3, 0.67, 0.43]
weights2 = [0.1, 0.45, 0.26, 0.72]
weights3 = [0.2, 0.73, 0.15, 0.61]

weights = [[0.5, 0.3, 0.67, 0.43],
		   [0.1, 0.45, 0.26, 0.72],
		   [0.2, 0.73, 0.15, 0.61]]

bias1 = 2
bias2 = 3
bias3 = 0.5

biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
	neuron_output = 0
	for input_neuron, weight_neuron in zip(inputs, neuron_weights):
		neuron_output += input_neuron*weight_neuron
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)

output = np.dot(weights, inputs) + biases

print(output)

output = np.dot(inputs1, np.array(weights).T) + biases

print(output)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(inputs1)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

activation = Activation_ReLU()

activation.forward(layer2.output)

print(activation.output)

# print(np.random.randn(4,3))

#---

inputs1 = [[1, 2, 3, 2.5],
		   [2, 5, -1, 2],
		   [-1.5, 2.7, 3.3, -0.8]]

layer1 = Layer_Dense(4,3)
layer2 = Layer_Dense(3,3)

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

layer1.forward(inputs1)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)
print(np.sum(activation2.output, axis=1))

inputs = [[-1.704815,1.503106,-0.817083,1.35356,-1.29366,1.362608,0.217795,0.462728,-1.237246],
		  [-0.888962,0.711473,-0.480023,0.748175,-0.981763,1.765482,-0.445018,1.719143,-1.09625]]

outputs = [[22.222923,-27.036774],
		   [18.88552,-28.37744]]

data_input = np.array(inputs)
data_output = np.array(outputs)

'''

file = open('inputs/ML-CUP22-TR.csv')

csvreader = csv.reader(file)
header = []
header = next(csvreader)
for i in range(6):
	next(csvreader)
train_X = []
train_Y = []
test_X = []
test_Y = []
for index, row in enumerate(csvreader):
	if index < 1000:
		train_X.append(row[1:10])
		train_Y.append(row[10 :])
	else:
		test_X.append(row[1:10])
		test_Y.append(row[10 :])

train_X = np.array(train_X, dtype=float)
train_Y = np.array(train_Y, dtype=float)
test_X = np.array(test_X, dtype=float)
test_Y = np.array(test_Y, dtype=float)

# pg = ParamConfig(1, 16, 1001, 0.5, 0.0001, 0.7)


# nn = NN1(9, 2, 2, 64)
# nn.gradient_descent(train_X, train_Y, test_X, test_Y, 5001, 1, 0.001, 0.7)

grid = Grid([1, 2],
			[16],
			[1001],
			[0.5],
			[0.0001],
			[0.7])

model = Model()
model.model_selection(train_X, train_Y, grid, CrossValidation(5))
model.model_assessment(test_X, test_Y)
model.print_model()