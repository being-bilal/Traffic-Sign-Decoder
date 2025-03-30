import numpy as np
import cv2
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from data import X_train, y_train, X_test, y_test
import joblib 


X_train = np.array(X_train)
X_test = np.array(X_test)

# Training the RandomForsest model
# n_estimators=100 : it is the number of desicion tress the model would create
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,  # Prevents trees from growing too deep
    min_samples_split=10,  # Avoids overfitting small samples
    min_samples_leaf=5,  # Ensures leaves contain multiple samples
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# saving the model
joblib.dump(model, "traffic_sign_rf_model.pkl", compress=3)
joblib.dump(X_test, "X_test.pkl", compress=3)
joblib.dump(y_test, "y_test.pkl", compress=3)
joblib.dump(X_train, "X_train.pkl", compress=3)
joblib.dump(y_train, "y_train.pkl", compress=3)