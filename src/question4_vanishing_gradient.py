import numpy as np
import wandb
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ann import MLP, get_optimizer


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_data():
    (X_train_full, y_train_full), _ = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    return X_train, X_val, y_train, y_val


def train_with_gradient_logging(activation_name, num_layers, X_train, X_val, y_train, y_val):
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    layer_sizes = [784] + [128] * num_layers + [10]
    model = MLP(layer_sizes, activation=activation_name, weight_init='xavier', loss='cross_entropy')
    optimizer = get_optimizer('adam', lr=0.001)

    wandb.init(project="da6401-assignment1",
               name=f"vanishing-grad-{activation_name}-{num_layers}layers",
               group="vanishing-gradient",
               reinit=True)

    for epoch in range(10):
        idx = np.random.permutation(len(X_train))
        X_s = X_train[idx]
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

        # Log gradient norm of first hidden layer
        first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)

        val_loss = model.compute_loss(model.forward(X_val), y_val_oh)
        val_acc  = np.mean(model.predict(X_val) == y_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "first_layer_grad_norm": first_layer_grad_norm,
        })

    wandb.finish()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data()

    configs = [
        ('sigmoid', 3),
        ('sigmoid', 5),
        ('relu', 3),
        ('relu', 5),
    ]

    for activation, layers in configs:
        print(f"Training {activation} with {layers} layers...")
        train_with_gradient_logging(activation, layers, X_train, X_val, y_train, y_val)
