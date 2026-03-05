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


def train_optimizer(optimizer_name, X_train, X_val, y_train, y_val):
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    # Fixed architecture: 3 hidden layers, 128 neurons, ReLU
    model = NeuralNetwork([784, 128, 128, 128, 10], activation='relu', weight_init='xavier')
    optimizer = get_optimizer(optimizer_name, lr=0.001)

    wandb.init(project="da6401-assignment1",
               name=f"optimizer-{optimizer_name}",
               group="optimizer-showdown",
               reinit=True)

    for epoch in range(10):
        idx = np.random.permutation(len(X_train))
        X_s = X_train[idx]
        y_s = y_train[idx]
        y_oh_s = y_train_oh[idx]

        total_loss = 0
        num_batches = 0
        for j in range(X_s.shape[0] // 64):
            Xb = X_s[j*64:(j+1)*64]
            yb = y_oh_s[j*64:(j+1)*64]
            probs = model.forward(Xb)
            total_loss += model.compute_loss(probs, yb)
            num_batches += 1
            model.backward(yb)
            optimizer.update(model.layers)

        val_loss = model.compute_loss(model.forward(X_val), y_val_oh)
        val_acc  = np.mean(model.predict(X_val) == y_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "optimizer": optimizer_name
        })

    wandb.finish()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data()
    optimizers = ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
    for opt in optimizers:
        print(f"Training with {opt}...")
        train_optimizer(opt, X_train, X_val, y_train, y_val)
