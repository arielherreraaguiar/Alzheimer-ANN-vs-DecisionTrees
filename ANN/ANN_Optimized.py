import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

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

# Step 3: Build the final model
def build_final_model(input_shape, activation='relu', dropout_rate=0.4, l2_rate=0.01):
    model = Sequential([
        Dense(32, activation=activation, input_shape=(input_shape,),
              kernel_regularizer=tf.keras.regularizers.l2(l2_rate)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Step 4: Train the model
def train_final_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
    model = build_final_model(X_train.shape[1], activation='relu', dropout_rate=0.4, l2_rate=0.01)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, verbose=1)
    return model, history

# Step 5: Save the model
def save_model(model, file_path='final_model.h5'):
    model.save(file_path)
    print(f"Model saved to {file_path}")

# Step 6: Plot performance
def plot_final_performance(history):
    # Losses
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Step 7: Evaluate and display metrics
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Main execution
if __name__ == "__main__":
    # File path to dataset
    filepath = 'alzheimers_disease_data.csv'

    # Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train final model
    final_model, final_history = train_final_model(X_train, y_train, X_val, y_val)

    # Save the model
    save_model(final_model, file_path='final_model.h5')

    # Plot performance
    plot_final_performance(final_history)

    # Evaluate the model
    evaluate_model(final_model, X_test, y_test)

