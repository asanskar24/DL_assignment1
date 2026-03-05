"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Relative path to saved model weights (.npy)")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")

    parser.add_argument("--hidden_layers", type=int, default=1,
                        help="Number of hidden layers")

    parser.add_argument("--num_neurons", type=int, default=128,
                        help="Number of neurons in each hidden layer")

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"],
                        help="Activation function")

    return parser.parse_args()


def load_model(model_path):
    weights = np.load(model_path, allow_pickle=True)
    return weights


def load_dataset(dataset_name):
    """
    Load dataset and preprocess.
    """
    if dataset_name == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test.reshape(-1, 784) / 255.0

    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    
    Returns dictionary containing:
    logits, loss, accuracy, f1, precision, recall
    """

    probs = model.forward(X_test)
    logits = probs
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    results = {
        "logits": logits,
        "loss": None,  # loss may not be required during inference
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return results


def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    # Load dataset
    X_test, y_test = load_dataset(args.dataset)

    # Define network architecture
    layer_sizes = [784] + [args.num_neurons] * args.hidden_layers + [10]

    # Build model
    model = NeuralNetwork(layer_sizes, activation=args.activation)

    # Load weights
    weights = load_model(args.model_path)

    # Apply weights
    model.set_weights(weights)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation complete!")

    return results


if __name__ == "__main__":
    main()