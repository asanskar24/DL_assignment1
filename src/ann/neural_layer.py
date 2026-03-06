"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import Activation


class Layer:
    """
    A single fully-connected layer in the MLP.
    Stores weights, biases, and all intermediate values needed for backprop.
    """

    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier'):
        """
        Args:
            input_size: number of input features
            output_size: number of neurons in this layer
            activation: activation function name ('relu', 'sigmoid', 'tanh', 'linear')
            weight_init: 'xavier' or 'random'
        """
        self.activation = activation

        # Weight initialization
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (input_size + output_size))
            self.W = np.random.randn(input_size, output_size) * scale
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # Set during forward pass — needed by backward pass
        self.input = None
        self.pre_activation = None  # Z = X @ W + b
        self.output = None          # A = activation(Z)

        # Gradients — set during backward pass, exposed for autograder
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward pass through this layer.
        Args:
            X: input of shape (batch_size, input_size)
        Returns:
            output of shape (batch_size, output_size)
        """
        self.input = X
        self.pre_activation = np.dot(self.input, self.W) + self.b
        self.output = Activation.activate(self.pre_activation, self.activation)
        return self.output

    def backward(self, delta):

        dZ = delta * Activation.derivative(self.pre_activation, self.activation)

        self.grad_W = np.dot(self.input.T, dZ) / self.input.shape[0]

        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / self.input.shape[0]

        return np.dot(dZ, self.W.T)