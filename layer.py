from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_pass(self):
        pass

    @abstractmethod
    def backward_pass(self):
        pass

# Fully Connected Layer
class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        # input_shape: number of input neurons
        # output_shape: number of output neurons

        self.weights = np.random.rand(input_shape, output_shape)
        self.bias = np.random.rand(1, output_shape)

        self.temp_weihgts = np.copy(self.weights)
        self.temp_bias = np.copy(self.bias)

    def save_weights(self):
        self.temp_weihgts = np.copy(self.weights)
        self.temp_bias = np.copy(self.bias)
    
    def restore_weights(self):
        self.weights = self.temp_weihgts
        self.bias = self.temp_bias

    def forward_pass(self, input_data):
        self.input_data = input_data
        self.output = np.dot(self.input_data, self.weights) + self.bias

        return self.output

    def backward_pass(self, output_error, learning_rate):
        # compute error
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input_data.T, output_error)
        bias_error = self.bias * output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error

# Activation Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.act = activation
        self.act_prime = activation_prime

    def forward_pass(self, input_data):
        self.input_data = input_data
        self.output = self.act(self.input_data)
        return self.output

    def backward_pass(self, output_error, learning_rate):
        return self.act_prime(self.input_data) * output_error
