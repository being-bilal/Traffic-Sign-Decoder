import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from preprocessing import preprocess_image  # Import your preprocessing function
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model
rf_model = joblib.load("assets/traffic_sign_rf_model.pkl")
y_train = joblib.load("assets/y_train.pkl")
X_train = joblib.load("assets/X_train.pkl")

# Set the test images folder path
test_folder = "dataset/Test"  # test image folder path

csv_path = "assets/Test.csv"  # Change to your actual CSV path
test_df = pd.read_csv(csv_path)

# Prepare test data
X_test = []
y_test = []

for _, row in test_df.iterrows():
    true_label = row["ClassId"]
    img_path = os.path.join("dataset", row["Path"])
    
    try:
        processed_img = preprocess_image(img_path)  # Applying preprocessing
        X_test.append(processed_img)
        y_test.append(true_label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Convert to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Finished processing all images.")

# Predict labels
y_predicted = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# classification report
report = classification_report(y_test, y_predicted)


# Confusion matrix 
cm = confusion_matrix(y_test, y_predicted)

# Accessing the important features
importances = rf_model.feature_importances_

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=False, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()
