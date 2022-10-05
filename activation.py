import numpy as np

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# linear activation function
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    x[x <= 0] = 0
    x[x > 1] = 1
    return x