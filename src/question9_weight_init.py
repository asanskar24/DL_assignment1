import numpy as np
import wandb
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .neural_network import NeuralNetwork
from .optimizers import get_optimizer


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_data():
    (X_train_full, y_train_full), _ = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    return X_train, X_val, y_train, y_val


class ZeroInitMLP(MLP):
    """MLP with all weights and biases initialized to zero."""
    def __init__(self, layer_sizes, activation='relu', loss='cross_entropy'):
        super().__init__(layer_sizes, activation=activation, loss=loss)
        # Override weights to zero after parent init
        for layer in self.layers:
            layer.W = np.zeros_like(layer.W)
            layer.b = np.zeros_like(layer.b)


def train_init_comparison(init_type, X_train, y_train):
    y_train_oh = one_hot(y_train)

    if init_type == 'zeros':
        model = ZeroInitMLP([784, 128, 128, 10], activation='relu')
    else:
        model = NeuralNetwork([784, 128, 128, 10], activation='relu', weight_init='xavier')

    optimizer = get_optimizer('adam', lr=0.001)

    wandb.init(project="da6401-assignment1",
               name=f"weight-init-{init_type}",
               group="weight-initialization",
               reinit=True)

    iteration = 0
    # Train for 50 iterations only (as per assignment)
    for j in range(50):
        Xb = X_train[j*64:(j+1)*64]
        yb = y_train_oh[j*64:(j+1)*64]

        model.forward(Xb)
        model.backward(yb)
        optimizer.update(model.layers)

        # Log gradients of 5 neurons in first hidden layer
        grad_log = {}
        for neuron_idx in range(5):
            grad_norm = float(np.linalg.norm(model.layers[0].grad_W[:, neuron_idx]))
            grad_log[f"neuron_{neuron_idx}_grad_norm"] = grad_norm

        wandb.log({"iteration": j + 1, **grad_log})

    wandb.finish()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data()

    print("Training with Zero initialization...")
    train_init_comparison('zeros', X_train, y_train)

    print("Training with Xavier initialization...")
    train_init_comparison('xavier', X_train, y_train)
