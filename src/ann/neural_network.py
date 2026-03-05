"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import Layer
from .objective_functions import Loss


class MLP:
    """
    Multi-Layer Perceptron built from Layer objects.
    Supports forward pass, loss computation, and backpropagation.
    """

    def __init__(self, layer_sizes, activation='relu', weight_init='xavier', loss='cross_entropy'):
        """
        Args:
            layer_sizes: list of sizes e.g. [784, 128, 128, 10]
            activation: activation function for hidden layers
            weight_init: 'xavier' or 'random'
            loss: 'cross_entropy' or 'mse'
        """
        self.loss_type = loss
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            # Last layer uses linear activation (softmax applied separately)
            act = activation if i < len(layer_sizes) - 2 else 'linear'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], act, weight_init))

        self.probs = None  # Stores softmax output after forward pass

    def softmax(self, Z):
        """Numerically stable softmax."""
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Full forward pass through all layers + softmax.
        Args:
            X: input of shape (batch_size, input_features)
        Returns:
            softmax probabilities of shape (batch_size, num_classes)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.probs = self.softmax(out)
        return self.probs

    def compute_loss(self, probs, y_onehot):
        """Compute scalar loss."""
        return Loss.compute(probs, y_onehot, self.loss_type)

    def backward(self, y_onehot):
        """
        Backpropagation through all layers.
        Args:
            y_onehot: one-hot encoded labels, shape (batch_size, num_classes)
        """
        # Initial gradient at output (cross-entropy + softmax combined gradient)
        delta = (self.probs - y_onehot) / y_onehot.shape[0]

        # Propagate gradient backward through each layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def predict(self, X):
        """Return predicted class indices."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)