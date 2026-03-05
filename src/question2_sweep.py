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
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    return X_train, X_val, y_train, y_val, X_test, y_test


# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs':        {'values': [5, 10]},
        'batch_size':    {'values': [32, 64, 128]},
        'optimizer':     {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'learning_rate': {'values': [0.1, 0.01, 0.001]},
        'num_layers':    {'values': [2, 3, 4]},
        'hidden_size':   {'values': [64, 128]},
        'activation':    {'values': ['relu', 'sigmoid', 'tanh']},
        'weight_init':   {'values': ['random', 'xavier']},
        'weight_decay':  {'values': [0.0, 0.0005]},
        'loss':          {'values': ['cross_entropy']},
    }
}


def run_sweep():
    wandb.init()
    config = wandb.config

    X_train, X_val, y_train, y_val, _, _ = load_data()
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    layer_sizes = [784] + [config.hidden_size] * config.num_layers + [10]
    model = MLP(layer_sizes, activation=config.activation,
                weight_init=config.weight_init, loss=config.loss)
    optimizer = get_optimizer(config.optimizer, config.learning_rate, config.weight_decay)

    for epoch in range(config.epochs):
        idx = np.random.permutation(len(X_train))
        X_train_s = X_train[idx]
        y_train_s = y_train[idx]
        y_train_oh_s = y_train_oh[idx]

        total_loss = 0
        num_batches = 0
        for j in range(X_train_s.shape[0] // config.batch_size):
            Xb = X_train_s[j * config.batch_size:(j + 1) * config.batch_size]
            yb = y_train_oh_s[j * config.batch_size:(j + 1) * config.batch_size]
            probs = model.forward(Xb)
            loss  = model.compute_loss(probs, yb)
            total_loss += loss
            num_batches += 1
            model.backward(yb)
            optimizer.update(model.layers)

        val_probs = model.forward(X_val)
        val_loss  = model.compute_loss(val_probs, y_val_oh)
        val_acc   = np.mean(model.predict(X_val) == y_val)
        train_acc = np.mean(model.predict(X_train_s) == y_train_s)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

    wandb.finish()


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="da6401-assignment1")
    wandb.agent(sweep_id, function=run_sweep, count=100)
