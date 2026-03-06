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
        """
        Parse command-line arguments for training.
        Compatible with autograder CLI.
        """

        parser = argparse.ArgumentParser(description="Train a neural network")

        # dataset
        parser.add_argument(
            "--dataset",
            type=str,
            default="mnist",
            choices=["mnist", "fashion_mnist"]
        )

        # training params
        parser.add_argument(
            "--epochs",
            type=int,
            default=10
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=64
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.001
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0
        )

        # optimizer
        parser.add_argument(
            "--optimizer",
            type=str,
            default="sgd",
            choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
        )

        # architecture
        parser.add_argument(
            "--num_layers",
            type=int,
            default=1,
            help="Number of hidden layers"
        )

        parser.add_argument(
            "--hidden_size",
            type=int,
            nargs="+",
            default=[128],
            help="Hidden layer sizes (list)"
        )

        parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            choices=["relu", "sigmoid", "tanh"]
        )

        # loss
        parser.add_argument(
            "--loss",
            type=str,
            default="cross_entropy",
            choices=["cross_entropy", "mse"]
        )

        # weight initialization
        parser.add_argument(
            "--weight_init",
            type=str,
            default="xavier",
            choices=["random", "xavier"]
        )

        # wandb
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="da6401_assignment"
        )

        # model save path
        parser.add_argument(
            "--model_save_path",
            type=str,
            default="best_model.npy"
        )

        args = parser.parse_args()

        # Ensure hidden_size matches num_layers if single value given
        if len(args.hidden_size) == 1 and args.num_layers > 1:
            args.hidden_size = args.hidden_size * args.num_layers

        return args


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