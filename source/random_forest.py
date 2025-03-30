import numpy as np
import cv2
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data import X_train, y_train, X_test, y_test
import joblib 

# Convert to grayscale
def extract_hog_features(image):
    """Convert color image to grayscale and extract HOG features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


X_train_features = [extract_hog_features(image) for image in X_train]
X_test_features  = [extract_hog_features(image) for image in X_test]

X_train_features = np.array(X_train_features)
X_test_features = np.array(X_test_features)

# Training the RandomForsest model
# n_estimators=100 : it is the number of desicion tress the model would create
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_features, y_train)

# saving the model
joblib.dump(model, "traffic_sign_rf_model.pkl", compress=3)
joblib.dump(X_test_features, "X_test.pkl", compress=3)
joblib.dump(y_test, "y_test.pkl", compress=3)


