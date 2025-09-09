import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and preprocess the data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Drop irrelevant columns
    data = data.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']


    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y

# Step 2: Split data into training, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 3: Build the neural network model
def build_model(input_shape, dropout_rate=0.3):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Step 4: Train and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history

# Step 5: Plot training history
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.show()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.show()

# Step 6: Evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Predictions and confusion matrix
    y_test_pred = model.predict(X_test)
    y_test_pred_binary = (y_test_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_test_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix for ANN Model')
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred_binary))

# Step 7: Save the trained model
def save_model(model, filepath):
    model.save(filepath)
    print(f"Model saved to {filepath}")

# Main execution
if __name__ == "__main__":
    # File path to dataset
    filepath = 'alzheimers_disease_data.csv'

    # Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Build model
    model = build_model(input_shape=X_train.shape[1], dropout_rate=0.3)

    # Train model
    history = train_and_evaluate_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, 'alzheimers_ann_model.h5')
