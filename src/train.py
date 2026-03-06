"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb

from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train neural network")

    parser.add_argument("-d","--dataset", default="mnist",
                        choices=["mnist","fashion_mnist"])

    parser.add_argument("-e","--epochs", type=int, default=10)

    parser.add_argument("-b","--batch_size", type=int, default=64)

    parser.add_argument("-lr","--learning_rate", type=float, default=0.001)

    parser.add_argument("-o","--optimizer", default="sgd",
                        choices=["sgd","momentum","nag","rmsprop","adam","nadam"])

    parser.add_argument("-nhl","--hidden_layers", type=int, default=1)

    parser.add_argument("-sz","--num_neurons", type=int, default=128)

    parser.add_argument("-a","--activation", default="relu",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("-l","--loss", default="cross_entropy")

    parser.add_argument("-wi","--weight_init", default="xavier")

    parser.add_argument("-wp","--wandb_project", default="da6401_assignment")

    parser.add_argument("--model_save_path", default="best_model.npy")

    return parser.parse_args()


# -----------------------------
# Data Loading
# -----------------------------

def load_data(dataset):

    if dataset == "mnist":
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
    else:
        (X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(-1,784)/255.0
    X_test = X_test.reshape(-1,784)/255.0

    return X_train,y_train,X_test,y_test


def one_hot(y, classes=10):
    return np.eye(classes)[y]


# -----------------------------
# Training
# -----------------------------

def main():

    args = parse_arguments()

    wandb.init(project=args.wandb_project, config=vars(args))

    X_train,y_train,X_test,y_test = load_data(args.dataset)

    X_train,X_val,y_train,y_val = train_test_split(
        X_train,y_train,test_size=0.1,random_state=42
    )

    y_train = one_hot(y_train)
    y_val_oh = one_hot(y_val)

    model = NeuralNetwork(args)

    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    best_val_acc = 0

    n = X_train.shape[0]

    for epoch in range(args.epochs):

        perm = np.random.permutation(n)

        X_train = X_train[perm]
        y_train = y_train[perm]

        for i in range(0,n,args.batch_size):

            Xb = X_train[i:i+args.batch_size]
            yb = y_train[i:i+args.batch_size]

            logits = model.forward(Xb)

            model.backward(yb,logits)

            optimizer.update(model.layers)

        val_logits = model.forward(X_val)

        preds = np.argmax(val_logits,axis=1)

        val_acc = np.mean(preds==y_val)

        wandb.log({
            "epoch":epoch,
            "val_accuracy":val_acc
        })

        print(f"Epoch {epoch+1}/{args.epochs} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            weights = model.get_weights()

            np.save(args.model_save_path,weights)

    wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()