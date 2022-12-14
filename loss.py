import numpy as np

def MSE(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))

def MSE_prime(y, y_pred):
    return (y_pred - y)
