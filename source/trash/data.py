import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from source.RandomForest.preprocessing import preprocess_image

# Set dataset directory
dataset_path = "dataset/test"  
categories = os.listdir(dataset_path)  

# Load images and labels
data = []
labels = []
image_size = (64, 64) 

for category in categories:
    class_path = os.path.join(dataset_path, category)
    class_index = categories.index(category)  # Assigning numerical values to label
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = preprocess_image(img_path)
            data.append(img)
            labels.append(class_index)


data = np.array(data, dtype="float32") 
#data /= 255.0 # normalising the value
labels = np.array(labels)


# Split into training (80%) and testing (20%)
# random_state=42 : ensures the dataset is split the same way every time such that The train & test labels will remain the same across multiple runs (42 is just a convention)
# stratify=label : ensures the dataset is split proportional to the labels. the class ratio (4:1) is preserved for every label for test and training 
#X_train, X_st, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels, shuffle=True)

# Checking the split
#print(f"Percentage of train data : {(len(X_train)/len(data)) * 100}")
#print(f"Percentage of test data : {(len(X_test)/len(data)) * 100}")
