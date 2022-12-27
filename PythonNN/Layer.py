import numpy as np

class Layer_Dense:
	#Initializing weights and biases
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.input = inputs
		self.output = np.dot(inputs, self.weights) + self.biases

	def backprop(self, gradient):
		self.weights_deriv = np.dot(self.input.T, gradient)
		self.biases_deriv = np.sum(gradient, axis=0, keepdims=True)

		# we calculate derivative of inputs to pass to the previous layer while back propagating
		self.inputs_deriv = np.dot(gradient, self.weights.T)

	def update_params(self, alpha):
		self.weights = self.weights - alpha * self.weights_deriv
		self.biases = self.biases - alpha * self.biases_deriv