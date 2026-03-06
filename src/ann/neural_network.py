import numpy as np
from .neural_layer import Layer
from .activations import Activation


class NeuralNetwork:

    def __init__(self, cli_args):

        self.activation = cli_args.activation
        self.num_layers = cli_args.hidden_layers
        self.hidden_size = cli_args.num_neurons
        self.lr = getattr(cli_args, "learning_rate", 0.001)

        layer_sizes = [784] + [self.hidden_size] * self.num_layers + [10]

        self.layers = []

        for i in range(len(layer_sizes) - 1):

            act = self.activation if i < len(layer_sizes) - 2 else "linear"

            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], act)
            )

    # -----------------------------
    # Forward Pass
    # -----------------------------

    def forward(self, X):

        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out  # logits

    # -----------------------------
    # Backward Pass
    # -----------------------------

    def backward(self, y_true, logits):

        batch_size = y_true.shape[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        delta = (probs - y_true) / batch_size

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):

            delta = layer.backward(delta)

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # -----------------------------
    # Weight Update
    # -----------------------------

    def update_weights(self):

        for layer in self.layers:

            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    # -----------------------------
    # Training Loop
    # -----------------------------

    def train(self, X_train, y_train, epochs=1, batch_size=32):

        n = X_train.shape[0]

        for epoch in range(epochs):

            perm = np.random.permutation(n)

            X_train = X_train[perm]
            y_train = y_train[perm]

            for i in range(0, n, batch_size):

                Xb = X_train[i:i + batch_size]
                yb = y_train[i:i + batch_size]

                logits = self.forward(Xb)

                self.backward(yb, logits)

                self.update_weights()

    # -----------------------------
    # Evaluation
    # -----------------------------

    def evaluate(self, X, y):

        logits = self.forward(X)

        preds = np.argmax(logits, axis=1)

        acc = np.mean(preds == y)

        return acc

    # -----------------------------
    # Weight Utilities (provided)
    # -----------------------------

    def get_weights(self):

        d = {}

        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        return d

    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()