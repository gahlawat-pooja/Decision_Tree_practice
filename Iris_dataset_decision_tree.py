# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest_basics import Node, RandomForest  # Ensure this module is available
from sklearn.model_selection import KFold
import gc  #garbage collector

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
model_dict= {}

def train_and_evaluate_decision_tree(X, y, k): 
    kf = KFold(n_splits=k, shuffle=True, random_state=0) # (random_state=32) to fix the seed value
    accuracy_scores = []

    for fold_i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the decision tree classifier
        clf = RandomForest(n_estimators = 5)
        clf.fit(X_train, y_train)
        model_dict[f'model_{fold_i}']=clf
        del clf
        gc.collect()
        # Make predictions on the test set
        y_pred = model_dict[f'model_{fold_i}'].predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(accuracy_scores)
    print(model_dict)
    return accuracy_scores

# Example usage:
# Assuming you have your feature matrix X and target variable y
accuracy_scores = train_and_evaluate_decision_tree(X, y, k=5)

# Print the accuracy scores for each fold
for i, accuracy in enumerate(accuracy_scores):
    print(f"Fold {i+1} Accuracy: {accuracy:.4f}")

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"Average Accuracy: {average_accuracy:.4f}")