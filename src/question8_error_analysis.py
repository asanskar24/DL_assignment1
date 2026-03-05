import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ann import NeuralNetwork, get_optimizer


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_data():
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full_flat = X_train_full.reshape(-1, 784) / 255.0
    X_test_flat = X_test.reshape(-1, 784) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_flat, y_train_full, test_size=0.1, random_state=42
    )
    return X_train, X_val, y_train, y_val, X_test_flat, X_test, y_test


def train_best_model(X_train, X_val, y_train, y_val):
    y_train_oh = one_hot(y_train)

    model = NeuralNetwork([784, 128, 128, 128, 10], activation='relu',
                weight_init='xavier', loss='cross_entropy')
    optimizer = get_optimizer('adam', lr=0.001)

    for epoch in range(15):
        idx = np.random.permutation(len(X_train))
        X_s = X_train[idx]
        y_oh_s = y_train_oh[idx]

        for j in range(X_s.shape[0] // 64):
            Xb = X_s[j*64:(j+1)*64]
            yb = y_oh_s[j*64:(j+1)*64]
            model.forward(Xb)
            model.backward(yb)
            optimizer.update(model.layers)

        val_acc = np.mean(model.predict(X_val) == y_val)
        print(f"Epoch {epoch+1}/15 | Val Acc: {val_acc:.4f}")

    return model


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, X_test_flat, X_test_raw, y_test = load_data()

    print("Training best model...")
    model = train_best_model(X_train, X_val, y_train, y_val)

    preds = model.predict(X_test_flat)

    wandb.init(project="da6401-assignment1", name="error-analysis")

    # --- Standard Confusion Matrix ---
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Best Model', fontsize=14)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

    # --- Creative Visualization: Most Confused Pairs ---
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    fig2, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig2.suptitle('Most Common Misclassifications', fontsize=14)

    # Find top 10 most confused pairs
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:
                confused_pairs.append((cm_no_diag[i, j], i, j))
    confused_pairs.sort(reverse=True)

    for idx, (count, true_label, pred_label) in enumerate(confused_pairs[:10]):
        # Find an example of this misclassification
        mask = (y_test == true_label) & (preds == pred_label)
        examples = np.where(mask)[0]
        if len(examples) > 0:
            img = X_test_raw[examples[0]]
            axes[idx//5][idx%5].imshow(img, cmap='gray')
            axes[idx//5][idx%5].set_title(f"True:{true_label} Pred:{pred_label}\n({count} times)", fontsize=9)
            axes[idx//5][idx%5].axis('off')

    plt.tight_layout()
    wandb.log({"misclassification_examples": wandb.Image(fig2)})
    plt.close()

    # Log overall accuracy
    acc = np.mean(preds == y_test)
    wandb.log({"test_accuracy": acc})
    print(f"Test Accuracy: {acc:.4f}")

    wandb.finish()
