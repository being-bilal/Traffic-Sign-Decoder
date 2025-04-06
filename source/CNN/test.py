import pandas as pd
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("source/CNN/cnn_model.keras")

# Load test data mapping
test_df = pd.read_csv("assets/Test.csv")
test_df['ClassId'] = test_df['ClassId'].astype(str)

# Load class mapping from mapping.json
try:
    with open("assets/mapping.json", "r") as f:
        mapping_json = json.load(f)
    print("Successfully loaded mapping.json")
except Exception as e:
    print(f"Error loading mapping.json: {e}")
    mapping_json = None

# Try to load class names if available from signnames.csv
try:
    signnames = pd.read_csv("assets/signnames.csv")
    class_dict = signnames.set_index('ClassId')['SignName'].to_dict()
    # Convert string keys to integers if needed
    class_names_csv = {int(k) if isinstance(k, str) else k: v for k, v in class_dict.items()}
    print("Successfully loaded signnames.csv")
except Exception as e:
    print(f"Error loading signnames.csv: {e}")
    class_names_csv = None

# Print first few rows of test_df to understand the class structure
print("\nFirst few rows of Test.csv:")
print(test_df.head())

# Check unique classes in the test dataset
unique_classes = test_df['ClassId'].unique()
print(f"\nUnique classes in test dataset: {len(unique_classes)}")
print(f"First few classes: {sorted(unique_classes)[:10]}")

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

# Check class indices mapping from the generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="dataset/",
    x_col="Path",
    y_col="ClassId",
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Get the class_indices from the generator
generator_class_indices = test_generator.class_indices
print("\nClass indices from generator:")
for class_name, idx in sorted(generator_class_indices.items(), key=lambda x: x[1]):
    print(f"Class '{class_name}' mapped to index {idx}")

# Compare with mapping.json if available
if mapping_json:
    print("\nComparing with mapping.json:")
    for i, (class_name, idx) in enumerate(sorted(generator_class_indices.items(), key=lambda x: x[1])):
        if i < 10:  # Just print first few for brevity
            json_value = mapping_json.get(class_name, "Not found")
            print(f"Class '{class_name}' -> Generator index: {idx}, mapping.json value: {json_value}")

# Compare with signnames.csv if available
if class_names_csv:
    print("\nComparing with signnames.csv:")
    for i, (class_name, idx) in enumerate(sorted(generator_class_indices.items(), key=lambda x: x[1])):
        if i < 10:  # Just print first few for brevity
            csv_name = class_names_csv.get(int(class_name) if class_name.isdigit() else class_name, "Not found")
            print(f"Class '{class_name}' -> Generator index: {idx}, CSV name: {csv_name}")

# Check a few examples with all mapping sources
test_generator.reset()
for i, (images, labels) in enumerate(test_generator):
    if i > 0:  # Just check the first batch
        break
        
    # Make predictions
    batch_pred = model.predict(images)
    batch_pred_classes = np.argmax(batch_pred, axis=1)
    
    # Calculate batch accuracy
    batch_acc = np.sum(batch_pred_classes == labels) / len(labels)
    print(f"\nExample batch accuracy: {batch_acc*100:.2f}%")
    
    # Convert generator indices back to original class names
    reverse_class_indices = {v: k for k, v in generator_class_indices.items()}
    
    # Show first few examples with detailed mapping
    print("\nDetailed prediction examples:")
    for j in range(min(5, len(labels))):
        true_label_idx = int(labels[j])
        pred_label_idx = int(batch_pred_classes[j])
        
        # Get original class names from generator indices
        true_class_name = reverse_class_indices.get(true_label_idx, f"Unknown-{true_label_idx}")
        pred_class_name = reverse_class_indices.get(pred_label_idx, f"Unknown-{pred_label_idx}")
        
        print(f"\nExample {j+1}:")
        print(f"  True label index: {true_label_idx}")
        print(f"  True class name (from generator): {true_class_name}")
        
        # Look up in mapping.json
        if mapping_json:
            true_json_value = mapping_json.get(true_class_name, "Not found")
            print(f"  True class meaning (from mapping.json): {true_json_value}")
        
        # Look up in signnames.csv
        if class_names_csv:
            true_csv_name = class_names_csv.get(int(true_class_name) if true_class_name.isdigit() else true_class_name, "Not found")
            print(f"  True class meaning (from signnames.csv): {true_csv_name}")
        
        print(f"  Predicted label index: {pred_label_idx}")
        print(f"  Predicted class name (from generator): {pred_class_name}")
        
        # Look up in mapping.json
        if mapping_json:
            pred_json_value = mapping_json.get(pred_class_name, "Not found")
            print(f"  Predicted class meaning (from mapping.json): {pred_json_value}")
        
        # Look up in signnames.csv
        if class_names_csv:
            pred_csv_name = class_names_csv.get(int(pred_class_name) if pred_class_name.isdigit() else pred_class_name, "Not found")
            print(f"  Predicted class meaning (from signnames.csv): {pred_csv_name}")
        
        # Display image with all information
        plt.figure(figsize=(8, 8))
        plt.imshow(images[j])
        
        # Construct title with all information
        title = f"True: {true_class_name}"
        if mapping_json and true_class_name in mapping_json:
            title += f"\n({mapping_json[true_class_name]})"
        
        title += f"\nPred: {pred_class_name}"
        if mapping_json and pred_class_name in mapping_json:
            title += f"\n({mapping_json[pred_class_name]})"
        
        title_color = 'green' if true_label_idx == pred_label_idx else 'red'
        plt.title(title, color=title_color)
        plt.axis('off')
        plt.show()
        