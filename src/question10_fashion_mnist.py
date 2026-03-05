import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .neural_network import NeuralNetwork
from .optimizers import get_optimizer


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_data():
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    return X_train, X_val, y_train, y_val, X_test, y_test


def train_fashion(config, X_train, X_val, y_train, y_val, X_test, y_test):
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    layer_sizes = [784] + [config['hidden_size']] * config['num_layers'] + [10]
    model = NeuralNetwork(layer_sizes, activation=config['activation'],
                weight_init='xavier', loss='cross_entropy')
    optimizer = get_optimizer(config['optimizer'], lr=config['lr'])

    wandb.init(project="da6401-assignment1",
               name=config['name'],
               group="fashion-mnist",
               reinit=True)

    for epoch in range(config['epochs']):
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

        val_loss = model.compute_loss(model.forward(X_val), y_val_oh)
        val_acc  = np.mean(model.predict(X_val) == y_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")

    test_acc = np.mean(model.predict(X_test) == y_test)
    wandb.log({"test_accuracy": test_acc})
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.finish()
    return test_acc


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, X_test, y_test = load_data()

    # 3 configs chosen based on MNIST learnings
    configs = [
        # Config 1: Best MNIST config — Adam + ReLU + Xavier
        {'name': 'fashion-adam-relu-3layers',  'optimizer': 'adam',  'activation': 'relu',
         'num_layers': 3, 'hidden_size': 128, 'lr': 0.001, 'epochs': 15},

        # Config 2: Nadam + Tanh — good for more complex datasets
        {'name': 'fashion-nadam-tanh-4layers', 'optimizer': 'nadam', 'activation': 'tanh',
         'num_layers': 4, 'hidden_size': 128, 'lr': 0.001, 'epochs': 15},

        # Config 3: Adam + ReLU but deeper — more capacity for complex patterns
        {'name': 'fashion-adam-relu-5layers',  'optimizer': 'adam',  'activation': 'relu',
         'num_layers': 5, 'hidden_size': 128, 'lr': 0.0005, 'epochs': 15},
    ]

    results = []
    for config in configs:
        print(f"\nRunning {config['name']}...")
        acc = train_fashion(config, X_train, X_val, y_train, y_val, X_test, y_test)
        results.append((config['name'], acc))

    print("\n--- Fashion-MNIST Results ---")
    for name, acc in results:
        print(f"{name}: {acc:.4f}")
