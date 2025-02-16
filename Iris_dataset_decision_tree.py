# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Weighted_hybrid_randomforest import RandomForest
from sklearn.model_selection import KFold
import gc  #garbage collector

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
model_dict= {}

def train_and_evaluate_random_forest(X, y, k): 
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    accuracy_scores = []

    for fold_i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the random forest classifier
        clf = RandomForest(max_depth=3, n_estimators=10, hybrid_split=True, weighted_voting=True)
        clf.fit(X_train, y_train)
        model_dict[f'model_{fold_i}'] = clf
        
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f"Fold {fold_i+1} Accuracy: {accuracy:.4f}")
        del clf
        gc.collect()
    
    return accuracy_scores

# Run training and evaluation
accuracy_scores = train_and_evaluate_random_forest(X, y, k=5)

# Print the accuracy scores for each fold
for i, accuracy in enumerate(accuracy_scores):
    print(f"Fold {i+1} Accuracy: {accuracy:.4f}")

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"Average Accuracy: {average_accuracy:.4f}")