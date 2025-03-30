import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("traffic_sign_rf_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

y_predicted = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# classification report
report = classification_report(y_test, y_predicted)


# Confusion matrix 
cm = confusion_matrix(y_test, y_predicted)

# Accessing the important features
importances = model.feature_importances_

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
