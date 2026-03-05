"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
import wandb
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann import NeuralNetwork, get_optimizer


def load_data(dataset_name):
    """Load and preprocess dataset."""
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test  = X_test.reshape(-1, 784) / 255.0
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    """Convert labels to one-hot encoding."""
    return np.eye(num_classes)[y]


def train(args):
    wandb.init(project="da6401-assignment1", config=vars(args))

    # Load and split data
    X_train_full, y_train_full, X_test, y_test = load_data(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    # Build model
    layer_sizes = [784] + [args.hidden_size] * args.num_layers + [10]
    model = NeuralNetwork(
        layer_sizes,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss
    )

    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)

    best_val_acc = 0

    for epoch in range(args.epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]
        y_train_oh = y_train_oh[idx]

        total_loss = 0
        num_batches = 0

        # Mini-batch training
        for j in range(X_train.shape[0] // args.batch_size):
            Xb = X_train[j * args.batch_size:(j + 1) * args.batch_size]
            yb = y_train_oh[j * args.batch_size:(j + 1) * args.batch_size]

            probs = model.forward(Xb)
            loss  = model.compute_loss(probs, yb)
            total_loss += loss
            num_batches += 1

            model.backward(yb)
            optimizer.update(model.layers)

        # Compute validation metrics
        val_probs = model.forward(X_val)
        val_loss  = model.compute_loss(val_probs, y_val_oh)
        val_preds = model.predict(X_val)
        val_acc   = np.mean(val_preds == y_val)

        train_preds = model.predict(X_train)
        train_acc   = np.mean(train_preds == y_train)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            np.save("best_model.npy", best_weights)

            config = {
                'dataset': args.dataset,
                'layer_sizes': layer_sizes,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'activation': args.activation,
                'weight_init': args.weight_init,
                'loss': args.loss,
                'optimizer': args.optimizer,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
            }
            with open('best_config.json', 'w') as f:
                json.dump(config, f, indent=2)

    wandb.finish()
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP on MNIST or Fashion-MNIST')
    parser.add_argument('-d',   '--dataset',       default='mnist',         help='mnist or fashion_mnist')
    parser.add_argument('-e',   '--epochs',        type=int,   default=10,  help='Number of epochs')
    parser.add_argument('-b',   '--batch_size',    type=int,   default=64,  help='Mini-batch size')
    parser.add_argument('-l',   '--loss',          default='cross_entropy', help='cross_entropy or mse')
    parser.add_argument('-o',   '--optimizer',     default='adam',          help='sgd/momentum/nag/rmsprop/adam/nadam')
    parser.add_argument('-lr',  '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd',  '--weight_decay',  type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers',    type=int,   default=3,   help='Number of hidden layers')
    parser.add_argument('-sz',  '--hidden_size',   type=int,   default=128, help='Neurons per hidden layer')
    parser.add_argument('-a',   '--activation',    default='relu',          help='sigmoid/tanh/relu')
    parser.add_argument('-wi',  '--weight_init',   default='xavier',        help='random or xavier')
    args = parser.parse_args()
    train(args)