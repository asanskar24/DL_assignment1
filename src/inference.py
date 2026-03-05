"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import json
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ann import NeuralNetwork


def load_test_data(dataset_name):
    """Load test data only."""
    if dataset_name == 'mnist':
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()
    return X_test.reshape(-1, 784) / 255.0, y_test
def load_model(weights_path, config_path=None):

    # Default architecture for dummy tests
    layer_sizes = [784, 128, 10]
    activation = "relu"
    weight_init = "xavier"
    loss = "cross_entropy"

    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)

        layer_sizes = config.get("layer_sizes", layer_sizes)
        activation = config.get("activation", activation)
        weight_init = config.get("weight_init", weight_init)
        loss = config.get("loss", loss)

    model = NeuralNetwork(
        layer_sizes,
        activation=activation,
        weight_init=weight_init,
        loss=loss
    )

    weights = np.load(weights_path, allow_pickle=True)

# handle object array saved with np.save
    if isinstance(weights, np.ndarray) and weights.shape == ():
        weights = weights.item()

    for i, layer in enumerate(model.layers):

        w = weights[i]

        if isinstance(w, dict):
            layer.W = w["W"]
            layer.b = w["b"]
        else:
            layer.W = w[0]
            layer.b = w[1]

    return model

def infer(args):
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load test data
    X_test, y_test = load_test_data(config['dataset'])

    # Rebuild model from config
    # Rebuild model from config
   
    model = load_model(args.weights, args.config)
    # Run predictions
    preds = model.predict(X_test)

    # Print metrics
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec  = recall_score(y_test, preds, average='macro')
    f1   = f1_score(y_test, preds, average='macro')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with saved MLP model')
    parser.add_argument('--weights', default='models/best_model.npy', help='Path to .npy weights file')
    parser.add_argument('--config',  default='models/best_config.json', help='Path to config JSON')
    args = parser.parse_args()
    infer(args)
