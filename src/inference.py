import argparse
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork


def parse_arguments():

    parser = argparse.ArgumentParser(description="Run inference")

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.npy"
    )

    parser.add_argument(
        "--dataset",
        default="mnist"
    )

    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_neurons",
        type=int,
        default=128
    )

    parser.add_argument(
        "--activation",
        default="relu"
    )

    return parser.parse_args()


def load_model(model_path):

    data = np.load(model_path, allow_pickle=True).item()

    return data


def load_dataset(dataset):

    if dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test.reshape(-1, 784) / 255.0

    return X_test, y_test


def evaluate_model(model, X_test, y_test):

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    return {
        "logits": logits,
        "loss": None,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def main():

    args = parse_arguments()

    X_test, y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)

    weights = load_model(args.model_path)

    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])


if __name__ == "__main__":
    main()