import numpy as np
import cv2
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_image  

# Load training data
train_csv = "assets/Train.csv" 
train_folder = "dataset" 

df = pd.read_csv(train_csv)

X_train = []
y_train = []

# Process all images
for _, row in df.iterrows():
    img_path = os.path.join(train_folder, row["Path"])
    label = row["ClassId"]
    
    try:
        processed_img = preprocess_image(img_path)  # Apply preprocessing
        X_train.append(processed_img)
        y_train.append(label)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the model
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15, max_features="sqrt")
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "assets/traffic_sign_rf_model.pkl", compress=3)
joblib.dump(X_train, "assets/X_train.pkl", compress=3)
joblib.dump(y_train, "assets/y_train.pkl", compress=3)

print("Model Training Complete!")
