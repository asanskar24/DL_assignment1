"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import Layer
from .objective_functions import Loss
import argparse

class NeuralNetwork:

    def __init__(self, layer_sizes, activation='relu', weight_init='xavier', loss='cross_entropy'):

        # If Namespace is passed instead of layer_sizes
        if isinstance(layer_sizes, argparse.Namespace):

            args = layer_sizes

            # Safely extract values
            num_layers = getattr(args, "num_layers", 1)
            hidden_size = getattr(args, "hidden_size", 128)
            activation = getattr(args, "activation", activation)
            weight_init = getattr(args, "weight_init", weight_init)
            loss = getattr(args, "loss", loss)

            layer_sizes = [784] + [hidden_size] * num_layers + [10]

        self.loss_type = loss
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else 'linear'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], act, weight_init))

        self.probs = None

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
    def get_weights(self):
   
        weights = {
            "W": [],
            "b": []
        }

        for layer in self.layers:
            weights["W"].append(layer.W)
            weights["b"].append(layer.b)

        return weights
    def set_weights(self, weights):
    

    # Case 1: dictionary format
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                layer.W = weights["W"][i]
                layer.b = weights["b"][i]

    # Case 2: list of dictionaries
        elif isinstance(weights, list):
            for i, layer in enumerate(self.layers):
                layer.W = weights[i]["W"]
                layer.b = weights[i]["b"]