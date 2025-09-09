import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and preprocess the data
file_path = 'alzheimers_disease_data.csv'
data = pd.read_csv(file_path)

# Drop irrelevant columns
data = data.drop(['PatientID', 'DoctorInCharge'], axis=1)

# Separate features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 2: Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Step 3: Pruning the Decision Tree with Cost-Complexity Pruning
def prune_decision_tree(X_train, y_train, X_val, y_val):
    # Train an initial decision tree to get the ccp_alpha values
    initial_tree = DecisionTreeClassifier(random_state=42)
    path = initial_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude the maximum alpha which would prune all nodes

    # Train trees for each alpha
    trees = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        trees.append(tree)

    # Evaluate each tree on the validation set
    val_scores = [accuracy_score(y_val, tree.predict(X_val)) for tree in trees]

    # Plot validation accuracy vs. ccp_alpha
    plt.figure(figsize=(8, 6))
    plt.plot(ccp_alphas, val_scores, marker='o', drawstyle="steps-post")
    plt.xlabel("Effective Alpha (ccp_alpha)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Effective Alpha (Pruning)")
    plt.show()

    # Select the tree with the highest validation accuracy
    best_alpha_index = np.argmax(val_scores)
    best_ccp_alpha = ccp_alphas[best_alpha_index]
    best_tree = trees[best_alpha_index]

    print(f"Best ccp_alpha: {best_ccp_alpha}")
    print(f"Best Validation Accuracy: {val_scores[best_alpha_index]}")
    
    return best_tree, best_ccp_alpha

# Main Execution
if __name__ == "__main__":
    # Prune the decision tree
    pruned_tree, best_ccp_alpha = prune_decision_tree(X_train, y_train, X_val, y_val)

    # Evaluate the pruned tree on the test set
    test_predictions = pruned_tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy of Pruned Tree: {test_accuracy}")

    # Classification Report
    print("\nClassification Report (Pruned Tree):")
    print(classification_report(y_test, test_predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No AD', 'AD'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Pruned Tree')
    plt.show()

    # Feature Importances
    plt.barh(X.columns, pruned_tree.feature_importances_)
    plt.title("Feature Importances - Pruned Tree")
    plt.xlabel("Importance Score")
    plt.show()

    # Visualize the pruned tree
    plt.figure(figsize=(20, 10))
    plot_tree(pruned_tree, feature_names=X.columns, class_names=['No AD', 'AD'], filled=True)
    plt.title("Decision Tree Visualization - Pruned Tree")
    plt.show()
