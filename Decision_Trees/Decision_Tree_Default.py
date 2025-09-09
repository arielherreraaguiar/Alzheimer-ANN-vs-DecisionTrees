import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Load and preprocess the dataset
file_path = 'alzheimers_disease_data.csv'
data = pd.read_csv(file_path)

# Drop irrelevant columns
data = data.drop(['PatientID', 'DoctorInCharge'], axis=1)

# Separate features (X) and target variable (y)
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Train the default Decision Tree model
default_model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='gini')
default_model.fit(X_train, y_train)

# Evaluate the model on validation and test sets
val_predictions = default_model.predict(X_val)
test_predictions = default_model.predict(X_test)

val_accuracy = default_model.score(X_val, y_val)
test_accuracy = default_model.score(X_test, y_test)

print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)

# Generate classification report and confusion matrix
print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_predictions))

# Plot confusion matrix
cm = confusion_matrix(y_test, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No AD', 'AD'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Default Model')
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(default_model, X, y, cv=5)
print(f"Cross-validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Plot cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', label='CV Accuracy')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean CV Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy - Default Model')
plt.legend()
plt.show()

# Plot the decision tree structure
plt.figure(figsize=(20, 10))
plot_tree(default_model, feature_names=X.columns, class_names=['No AD', 'AD'], filled=True)
plt.title('Decision Tree Visualization - Default Model')
plt.show()

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, default_model.feature_importances_)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances - Default Model')
plt.show()

# Save the trained model for future use
joblib.dump(default_model, 'default_decision_tree_model.pkl')
print("Default model saved as 'default_decision_tree_model.pkl'")
