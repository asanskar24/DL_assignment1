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


def train_dead_neuron_analysis(activation_name, learning_rate, X_train, X_val, y_train, y_val):
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    model = MLP([784, 128, 128, 128, 10], activation=activation_name,
                weight_init='xavier', loss='cross_entropy')
    optimizer = get_optimizer('sgd', lr=learning_rate)

    wandb.init(project="da6401-assignment1",
               name=f"dead-neuron-{activation_name}-lr{learning_rate}",
               group="dead-neuron",
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

        # Check dead neurons in first hidden layer
        # Run a batch through and check activations
        sample_probs = model.forward(X_val[:500])
        hidden_activations = model.layers[0].output  # activations after first layer

        if activation_name == 'relu':
            dead_neurons = np.mean(hidden_activations == 0, axis=0)  # fraction of time each neuron is 0
            dead_count = np.sum(np.mean(hidden_activations == 0, axis=0) > 0.99)
        else:
            dead_neurons = np.zeros(128)
            dead_count = 0

        val_loss = model.compute_loss(model.forward(X_val), y_val_oh)
        val_acc  = np.mean(model.predict(X_val) == y_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "dead_neuron_count": int(dead_count),
            "mean_activation": float(np.mean(hidden_activations)),
        })

    wandb.finish()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data()

    # ReLU with high learning rate (will cause dead neurons)
    print("ReLU with high learning rate (0.1)...")
    train_dead_neuron_analysis('relu', 0.1, X_train, X_val, y_train, y_val)

    # ReLU with normal learning rate
    print("ReLU with normal learning rate (0.01)...")
    train_dead_neuron_analysis('relu', 0.01, X_train, X_val, y_train, y_val)

    # Tanh with high learning rate (no dead neurons)
    print("Tanh with high learning rate (0.1)...")
    train_dead_neuron_analysis('tanh', 0.1, X_train, X_val, y_train, y_val)
