import numpy as np

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
		self.biases = np.zeros((1, n_neurons))
		print(self.weights.shape)
		print(self.biases.shape)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
