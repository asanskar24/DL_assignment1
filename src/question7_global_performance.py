import numpy as np
import wandb
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ann import NeuralNetwork, get_optimizer


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


def run_config(config, X_train, X_val, y_train, y_val, X_test, y_test):
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    layer_sizes = [784] + [config['hidden_size']] * config['num_layers'] + [10]
    model = NeuralNetwork(layer_sizes, activation=config['activation'],
                weight_init='xavier', loss='cross_entropy')
    optimizer = get_optimizer(config['optimizer'], lr=config['lr'])

    wandb.init(project="da6401-assignment1",
               name=config['name'],
               group="global-performance",
               reinit=True)

    for epoch in range(config['epochs']):
        idx = np.random.permutation(len(X_train))
        X_s = X_train[idx]
        y_oh_s = y_train_oh[idx]

        for j in range(X_s.shape[0] // 64):
            Xb = X_s[j*64:(j+1)*64]
            yb = y_oh_s[j*64:(j+1)*64]
            probs = model.forward(Xb)
            model.backward(yb)
            optimizer.update(model.layers)

    # Final metrics
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc  = np.mean(model.predict(X_test) == y_test)

    wandb.log({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "gap": train_acc - test_acc,
        "optimizer": config['optimizer'],
        "activation": config['activation'],
        "num_layers": config['num_layers'],
    })

    print(f"{config['name']} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}")
    wandb.finish()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, X_test, y_test = load_data()

    # Variety of configs to show overfitting vs good generalization
    configs = [
        {'name': 'adam-relu-3layers',    'optimizer': 'adam',     'activation': 'relu',    'num_layers': 3, 'hidden_size': 128, 'lr': 0.001, 'epochs': 10},
        {'name': 'adam-relu-6layers',    'optimizer': 'adam',     'activation': 'relu',    'num_layers': 6, 'hidden_size': 128, 'lr': 0.001, 'epochs': 10},
        {'name': 'sgd-sigmoid-3layers',  'optimizer': 'sgd',      'activation': 'sigmoid', 'num_layers': 3, 'hidden_size': 128, 'lr': 0.01,  'epochs': 10},
        {'name': 'adam-tanh-4layers',    'optimizer': 'adam',     'activation': 'tanh',    'num_layers': 4, 'hidden_size': 128, 'lr': 0.001, 'epochs': 10},
        {'name': 'nadam-relu-3layers',   'optimizer': 'nadam',    'activation': 'relu',    'num_layers': 3, 'hidden_size': 64,  'lr': 0.001, 'epochs': 10},
        {'name': 'rmsprop-relu-2layers', 'optimizer': 'rmsprop',  'activation': 'relu',    'num_layers': 2, 'hidden_size': 128, 'lr': 0.001, 'epochs': 10},
    ]

    for config in configs:
        print(f"Running {config['name']}...")
        run_config(config, X_train, X_val, y_train, y_val, X_test, y_test)
