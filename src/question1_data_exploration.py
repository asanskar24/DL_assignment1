import numpy as np
import wandb
from keras.datasets import mnist

# Initialize wandb
wandb.init(project="da6401-assignment1", name="data-exploration")

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0  # normalize but keep 28x28 shape for visualization

# Class names for MNIST
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Create W&B Table with 5 samples from each class
table = wandb.Table(columns=["image", "label", "class_name"])

for class_label in range(10):
    indices = np.where(y_train == class_label)[0][:5]
    for idx in indices:
        image = X_train[idx]  # already 28x28
        table.add_data(wandb.Image(image), class_label, class_names[class_label])

wandb.log({"class_samples": table})

print("Logged 50 sample images (5 per class) to W&B Table")

# Log class distribution as a bar chart
class_counts = [np.sum(y_train == i) for i in range(10)]
distribution_table = wandb.Table(columns=["class", "count"])
for i, count in enumerate(class_counts):
    distribution_table.add_data(class_names[i], count)

wandb.log({"class_distribution": distribution_table})

print("Class distribution:")
for i, count in enumerate(class_counts):
    print(f"  Class {i}: {count} samples")

wandb.finish()
