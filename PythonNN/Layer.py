import numpy as np
from scipy import signal


class Layer:
    #Initializing weights and biases
    def __init__(self, n_inputs, n_neurons, activ_function=None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def forward(self, inputs):
        pass
        

    def backprop(self, gradient):
        pass

    def update(self, optimizer):
        pass



class Dense(Layer):
    #Initializing weights and biases
    def __init__(self, n_inputs, n_neurons, activ_function=None):
        limit = 0.05
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
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

    def reset_params(self):
        limit = 0.05
        self.weights = np.random.uniform(-limit, limit, size=(self.n_inputs,self.n_neurons))
        self.biases = np.zeros((1, self.n_neurons))
        self.momentums_weight = np.zeros_like(self.weights)
        self.momentums_bias = np.zeros_like(self.biases)

    def update(self, optimizer):
        self.weights -= optimizer.lr * self.weights_deriv
        self.biases -= optimizer.lr * self.biases_deriv

# Convolutional Layer
class ConvolutionalLayer(Layer):
    
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.inputs[j], self.kernels[i,j], "valid")

    def backprop(self, gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.inputs[j], gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.kernels[i, j], "full")

        self.kernels_deriv = kernels_gradient
        self.biases_deriv = gradient
        self.inputs_deriv = input_gradient


    def update(self, optimizer):
        self.kernels -= optimizer.lr * self.kernels_deriv
        self.biases -= optimizer.lr * self.biases_deriv


# Reshape Layer
class ReshapeLayer(Layer):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs):
        self.output = np.reshape(inputs, self.output_shape)

    def backprop(self, gradient):
        self.inputs_deriv = np.reshape(gradient, self.input_shape)


class MaxPooling(Layer):
    def __init__(self, input_shape, ksize=(2,2)):
        self.ksize = ksize
        self.input_shape = input_shape

    def forward(self, input):
        self.input = input
        self.output = pooling(input, self.ksize, pad=True)

    def backprop(self, gradient):
        # inputs deriv has the shape of the inputs.
        # for each region, the derivative for the maximum element is 1, and the rest is 0.
        # we have to multiply each value by the gradient, therefore the maximum element will
        # have the value of the gradient in that spot.
        self.inputs_deriv = np.zeros(self.input_shape)
        depth = self.input_shape[0] # input depth
        height_out = self.output.shape[0]
        width_out = self.output.shape[1]
        pool_height = self.ksize[0]
        pool_width = self.ksize[1]
        for d in range(depth):
            for h in range(height_out):
                for w in range(width_out):
                    h_max, w_max = np.where(np.max(self.input[d, h*pool_height : (h+1)*pool_height, w*pool_width : (w+1)*pool_width]) \
                       == self.input[d, h*pool_height : (h+1)*pool_height, w*pool_width : (w+1)*pool_width])
                    h_max = h_max[0]
                    w_max = w_max[0]
                    self.inputs_deriv[d, h*pool_height : (h+1)*pool_height, w*pool_width : (w+1)*pool_width][h_max][w_max] \
                        = gradient[d, h, w]

def pooling(mat,ksize,method='max',pad=False):
    """
    Non-overlapping pooling on 2D or 3D data
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[1:] # ignore the number of inputs
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=mat.shape[:1]+(ny*ky, nx*kx)
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=mat.shape[:1]+(ny,ky,nx,kx)

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(2,4))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(2,4))

    return result