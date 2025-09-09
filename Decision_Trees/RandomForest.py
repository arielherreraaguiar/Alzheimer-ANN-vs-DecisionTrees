import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib

# Load and preprocess the data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Drop irrelevant columns
    data = data.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = (X[numerical_cols] - X[numerical_cols].mean()) / X[numerical_cols].std()

    return X, y

# Train and evaluate Random Forest
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    # Define the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,           # Number of trees
        max_depth=5,                # Maximum depth of each tree
        min_samples_split=2,        # Minimum samples to split
        min_samples_leaf=5,         # Minimum samples per leaf
        random_state=42,            # For reproducibility
        criterion='gini'            # Criterion for splitting
    )

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Evaluate on validation data
    val_predictions = rf_model.predict(X_val)
    val_accuracy = rf_model.score(X_val, y_val)

    # Evaluate on test data
    test_predictions = rf_model.predict(X_test)
    test_accuracy = rf_model.score(X_test, y_test)

    # Cross-validation accuracy
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    mean_cv_accuracy = cv_scores.mean()

    print("Validation Accuracy (Random Forest):", val_accuracy)
    print("Test Accuracy (Random Forest):", test_accuracy)
    print("Cross-validation Accuracy Scores:", cv_scores)
    print("Mean Cross-validation Accuracy:", mean_cv_accuracy)

    # Classification report
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, test_predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No AD', 'AD'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # Feature importances
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importances - Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()

    # Save the model
    model_file = 'random_forest_model.pkl'
    joblib.dump(rf_model, model_file)
    print(f"Random Forest model saved as {model_file}")

    return rf_model

# Main execution
if __name__ == "__main__":
    # File path to dataset
    filepath = 'alzheimers_disease_data.csv'

    # Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
