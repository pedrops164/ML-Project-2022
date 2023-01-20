import numpy as np

class Layer:
	#Initializing weights and biases
	def __init__(self, n_inputs, n_neurons, activ_function=None):
		#self.weights = 0.05 * np.random.randn(n_inputs,n_neurons)
<<<<<<< Updated upstream
		limit = 0.1
		self.n_inputs = n_inputs
		self.n_neurons = n_neurons
		#self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
=======
		limit = 0.7
		self.n_inputs = n_inputs
		self.n_neurons = n_neurons
		#self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
>>>>>>> Stashed changes
		self.weights = np.random.uniform(-limit, limit, size=(n_inputs,n_neurons))
		self.biases = np.zeros((1, n_neurons))
		self.activ = activ_function
		self.momentums_weight = np.zeros_like(self.weights)
		self.momentums_bias = np.zeros_like(self.biases)

	def forward(self, inputs):
		self.input = inputs
		Z1 = np.dot(inputs, self.weights) + self.biases
		if self.activ == None:
			self.output = Z1
		else:
			self.activ.forward(Z1)
			A1 = self.activ.output
			self.output = A1
		

	def backprop(self, gradient):
		if self.activ != None:
			self.activ.backprop(gradient)
			gradient = self.activ.inputs_deriv

		self.weights_deriv = np.dot(self.input.T, gradient)
		self.biases_deriv = np.sum(gradient, axis=0, keepdims=True)

		# we calculate derivative of inputs to pass to the previous layer while back propagating
		self.inputs_deriv = np.dot(gradient, self.weights.T)




	# def update_params(self, alpha):
	# 	self.weights = self.weights - alpha * self.weights_deriv
	# 	self.biases = self.biases - alpha * self.biases_deriv

	def reset_params(self):
		limit = 0.1
		self.weights = np.random.uniform(-limit, limit, size=(self.n_inputs,self.n_neurons))
		self.biases = np.zeros((1, self.n_neurons))
		self.momentums_weight = np.zeros_like(self.weights)
		self.momentums_bias = np.zeros_like(self.biases)