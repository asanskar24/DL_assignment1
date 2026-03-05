import numpy as np


class Activation:
   

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    @staticmethod
    def sigmoid_derivative(Z):
        s = Activation.sigmoid(Z)
        return s *(1 - s)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.tanh(Z) ** 2

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def linear(Z):
        return Z

    @staticmethod
    def linear_derivative(Z):
        return np.ones_like(Z)

    @staticmethod
    def activate(Z, name):
        
        functions = {
            'sigmoid': Activation.sigmoid,
            'tanh': Activation.tanh,
            'relu': Activation.relu,
            'linear': Activation.linear
        }
        return functions[name](Z)

    @staticmethod
    def derivative(Z, name):
        
        derivatives = {
            'sigmoid': Activation.sigmoid_derivative,
            'tanh': Activation.tanh_derivative,
            'relu': Activation.relu_derivative,
            'linear': Activation.linear_derivative
        }
        return derivatives[name](Z)