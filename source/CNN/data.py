import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
import tensorflow as tf
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from preprocess import get_data_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Getting training and validation generators
train_dir = "dataset/"
train_generator, validation_generator = get_data_generators(train_dir)
class_names = list(train_generator.class_indices.keys())

# Get training distribution
train_labels = train_generator.classes
train_unique, train_counts = np.unique(train_labels, return_counts=True)

# Get validation distribution
val_labels = validation_generator.classes
val_unique, val_counts = np.unique(val_labels, return_counts=True)

# Align class names with counts
train_class_counts = dict(zip(train_unique, train_counts))
val_class_counts = dict(zip(val_unique, val_counts))

# Load mapping from JSON
with open("assets/mapping.json", "r") as f:
    class_labels = json.load(f)

# Create label names using the mapping


# Bar plot
x = np.arange(len(class_names))
width = 0.4

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, [train_class_counts.get(i, 0) for i in range(len(class_names))],
        width, label='Train')
plt.bar(x + width/2, [val_class_counts.get(i, 0) for i in range(len(class_names))],
        width, label='Validation')

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Train and Validation Sets")
plt.xticks(x, class_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
