import joblib
import cv2
import numpy as np
import json
from preprocessing import preprocess_image 
import matplotlib.pyplot as plt


rf_model = joblib.load("assets/traffic_sign_rf_model.pkl")

with open("assets/mapping.json", "r") as f:
    class_labels = json.load(f)
    
image_path = "dataset/Test/00044.png"

# Preprocessing the image
processed_img = preprocess_image(image_path)

# Reshaping (model expects 2D input)
processed_img = processed_img.reshape(1, -1)

predicted_class = rf_model.predict(processed_img)[0]

sign_name = class_labels.get(str(predicted_class), "Unknown Sign")

# Print the result
print(f"Predicted Class ID: {predicted_class}")
print(f"Predicted Traffic Sign: {sign_name}")


img = cv2.imread(image_path)
predicted_img = cv2.imread(f"dataset/Meta/{str(predicted_class)}.png") 

plt.figure(figsize=(8, 4))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Processed Image
plt.subplot(1, 2, 2)
plt.imshow(predicted_img)  
plt.title("Predicted Image")
plt.axis("off")

plt.suptitle(f"Predicted Sign: {sign_name}", fontsize=14, fontweight="bold")
plt.show()
