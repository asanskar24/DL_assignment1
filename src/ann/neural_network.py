import numpy as np
import argparse
from .neural_layer import Layer
from .objective_functions import Loss


class NeuralNetwork:
    """
    Multi-layer perceptron neural network.
    """

    def __init__(self, layer_sizes, activation='relu', weight_init='xavier', loss='cross_entropy'):

        # Handle argparse Namespace
        if isinstance(layer_sizes, argparse.Namespace):
            args = layer_sizes
            num_layers = getattr(args, "num_layers", 1)
            hidden_size = getattr(args, "hidden_size", 128)

            activation = getattr(args, "activation", activation)
            weight_init = getattr(args, "weight_init", weight_init)
            loss = getattr(args, "loss", loss)

            layer_sizes = [784] + [hidden_size] * num_layers + [10]

        self.loss_type = loss
        self.layers = []

        # Build layers
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else "linear"
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], act, weight_init)
            )

        self.logits = None

    # -------------------------
    # Utility functions
    # -------------------------

    def softmax(self, Z):
        """Numerically stable softmax."""
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # -------------------------
    # Forward Pass
    # -------------------------

    def forward(self, X):
        """
        Forward pass through the network.
        Returns logits (no softmax applied).
        """
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        self.logits = out
        return out

    # -------------------------
    # Loss
    # -------------------------

    def compute_loss(self, logits, y_onehot):
        probs = self.softmax(logits)
        return Loss.compute(probs, y_onehot, self.loss_type)

    # -------------------------
    # Backpropagation
    # -------------------------

    def backward(self, logits, y_onehot):

        probs = self.softmax(logits)

        delta = (probs - y_onehot) / y_onehot.shape[0]

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    # -------------------------
    # Prediction
    # -------------------------

    def predict(self, X):

        logits = self.forward(X)
        probs = self.softmax(logits)

        return np.argmax(probs, axis=1)

    # -------------------------
    # Weight Utilities
    # -------------------------

    def get_weights(self):

        weights = {"W": [], "b": []}

        for layer in self.layers:
            weights["W"].append(layer.W)
            weights["b"].append(layer.b)

        return weights

    def set_weights(self, weights):
  
        if isinstance(weights, np.ndarray):
            weights = weights.tolist()

        # Format 1: {"W":[...], "b":[...]}
        if isinstance(weights, dict) and "W" in weights:
            for i, layer in enumerate(self.layers):
                layer.W = weights["W"][i]
                layer.b = weights["b"][i]

        # Format 2: [(W,b), (W,b)]
        elif isinstance(weights, list):
            for i, layer in enumerate(self.layers):

                w = weights[i]

                if isinstance(w, tuple):
                    layer.W = w[0]
                    layer.b = w[1]

                elif isinstance(w, dict):
                    layer.W = w["W"]
                    layer.b = w["b"]

                else:
                    raise ValueError("Unknown weight format")