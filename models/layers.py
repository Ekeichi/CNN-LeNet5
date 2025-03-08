import numpy as np
from scipy import signal

class Convolutional():
    def __init__(self, input_shape, kernel_size, depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        self.padding = padding

        # Dimensions de l'entrée
        self.batch_size, self.input_depth, self.input_height, self.input_width = input_shape

        # Initialisation des noyaux (kernels) et des biais
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (self.input_depth * kernel_size * kernel_size))
        self.biases = np.zeros(self.depth)

        # Calcul de la taille de la sortie (output)
        self.output_height = (self.input_height - kernel_size + 2 * padding) // stride + 1
        self.output_width = (self.input_width - kernel_size + 2 * padding) // stride + 1
        self.output_shape = (self.batch_size, self.depth, self.output_height, self.output_width)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        for b in range(self.batch_size):
            for d in range(self.depth):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        input_patch = self.input[b, :, h_start:h_end, w_start:w_end]
                        self.output[b, d, i, j] = np.sum(input_patch * self.kernels[d]) + self.biases[d]
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        # Batch-wise processing
        for b in range(self.batch_size):
        # Compute kernel gradients
            for d in range(self.depth):
                for j in range(self.input_depth):
                    # Use correlate for kernel gradient computation
                    kernels_gradient[d, j] += signal.correlate2d(self.input[b, j], output_gradient[b, d], mode="valid")

        # Compute input gradients
            for d in range(self.depth):
                for j in range(self.input_depth):
                # Use convolution for input gradient
                    input_gradient[b, j] += signal.convolve2d(output_gradient[b, d], self.kernels[d, j], mode="full")

        # Update parameters
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 2, 3))
        return input_gradient
    

class Fc():
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / input_shape)
        self.bias = np.random.randn(1, output_shape)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias
        
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)
        return input_gradient

class avg_pooling():
    def forward(self, input, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        self.input = input

        batch, depth_input, height, width = input.shape

        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1

        output = np.zeros((batch, depth_input,output_height, output_width))

        for b in range(0, batch):
            for d in range(0, depth_input):
                for h in range(0, output_height):
                    for w in range(0, output_width):
                        start_h = h*stride
                        start_w = w*stride
                        end_h = start_h + pool_size
                        end_w = start_w + pool_size

                        window = input[b, d, start_h:end_h , start_w:end_w]
                        output[b, d, h, w] = np.mean(window)
        return output
    
    def backward(self, d_out):
    # Dimensions du tenseur d'entrée
        stride = self.stride
        pool_size = self.pool_size 

        B, C, H_in, W_in = self.input.shape
        _, _, H_out, W_out = d_out.shape

        d_input = np.zeros_like(self.input)

        pool_area = pool_size * pool_size

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):

                        start_i = i * stride
                        start_j = j * stride
                        end_i = start_i + pool_size
                        end_j = start_j + pool_size

                        d_input[b, c, start_i:end_i, start_j:end_j] += d_out[b, c, i, j] / pool_area
        return d_input
                    
class Relu():
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient):
        return output_gradient * np.where(self.input > 0, 1, 0)
    
class sigmoid():
    def forward(self, input):
        self.input = input
        self.output = 1/(1 + np.exp(-input))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)

class tanh():
    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * (1 - self.output**2)