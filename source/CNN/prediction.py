import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import get_data_generators

# Load the model
model = load_model("source/CNN/cnn_model.keras")  # Change filename if needed
train_dir = "dataset/"
train_generator, validation_generator = get_data_generators(train_dir)

x_batch, y_batch = next(iter(train_generator))

# Make predictions
predictions = model.predict(x_batch)
pred_classes = np.argmax(predictions, axis=1)

# Compare with actual labels
correct = np.sum(pred_classes == y_batch)
print(f"Batch accuracy: {correct/len(y_batch)*100:.2f}%")

# Check a few examples
for i in range(5):
    print(f"True: {y_batch[i]}, Predicted: {pred_classes[i]}")