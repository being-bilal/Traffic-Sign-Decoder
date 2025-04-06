import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from preprocessing import preprocess_image  
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Loading the trained model
rf_model = joblib.load("assets/traffic_sign_rf_model.pkl")
y_train = joblib.load("assets/y_train.pkl")
X_train = joblib.load("assets/X_train.pkl")

test_folder = "dataset/Test" 
csv_path = "assets/Test.csv"  
test_df = pd.read_csv(csv_path)

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


X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict labels
y_predicted = rf_model.predict(X_test)

# Calculating accuracy and creating classification report
accuracy = accuracy_score(y_test, y_predicted)
print(f"Model Accuracy: {accuracy*100:.2f}%")
report = classification_report(y_test, y_predicted)
print(report)

# Confusion matrix 
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=False, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
