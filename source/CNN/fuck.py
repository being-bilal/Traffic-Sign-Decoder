import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model("source/CNN/cnn_model.keras")

# Load test data mapping
test_df = pd.read_csv("assets/Test.csv")
test_df['ClassId'] = test_df['ClassId'].astype(str)


# Setup test data generator with the same preprocessing as training
test_datagen = ImageDataGenerator(        
        rescale=1./255,
        rotation_range=10,      
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        shear_range=0.1
        )  

# Make sure to use the same target_size as during training
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="dataset/",
    x_col="Path",  # Adjust column name if different
    y_col="ClassId",  # Adjust column name if different
    target_size=(64, 64),  # Must match your training input size
    batch_size=32,
    class_mode='sparse',
    shuffle=False  # Important: keep false to maintain order for confusion matrix
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Make predictions
test_generator.reset()
y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes

# Clip predictions to match actual number of samples
# (last batch might have fewer samples)
y_pred_classes = y_pred_classes[:len(y_true)]

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('test_confusion_matrix.png')
plt.show()

# Check a few examples
test_generator.reset()
for i, (images, labels) in enumerate(test_generator):
    if i > 0:  # Just check the first batch
        break
        
    # Make predictions
    batch_pred = model.predict(images)
    batch_pred_classes = np.argmax(batch_pred, axis=1)
    
    # Calculate batch accuracy
    batch_acc = np.sum(batch_pred_classes == labels) / len(labels)
    print(f"Example batch accuracy: {batch_acc*100:.2f}%")
    
    # Show first few examples
    for j in range(min(5, len(labels))):
        print(f"True: {int(labels[j])}, Predicted: {batch_pred_classes[j]}")