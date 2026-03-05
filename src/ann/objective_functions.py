import numpy as np


class Loss:
    """
    Loss functions for training the MLP.
    All functions return a scalar loss value.
    """

    @staticmethod
    def cross_entropy(probs, y_onehot):
        """
        Cross-entropy loss for multi-class classification.
        Args:
            probs: softmax probabilities, shape (batch_size, num_classes)
            y_onehot: one-hot encoded labels, shape (batch_size, num_classes)
        Returns:
            scalar loss
        """
        return -np.sum(y_onehot * np.log(probs + 1e-8)) / probs.shape[0]

    @staticmethod
    def mse(probs, y_onehot):
        """
        Mean squared error loss.
        Args:
            probs: model output, shape (batch_size, num_classes)
            y_onehot: one-hot encoded labels, shape (batch_size, num_classes)
        Returns:
            scalar loss
        """
        return np.mean((probs - y_onehot) ** 2)

    @staticmethod
    def compute(probs, y_onehot, loss_type='cross_entropy'):
        """Compute loss by name."""
        if loss_type == 'cross_entropy':
            return Loss.cross_entropy(probs, y_onehot)
        elif loss_type == 'mse':
            return Loss.mse(probs, y_onehot)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")