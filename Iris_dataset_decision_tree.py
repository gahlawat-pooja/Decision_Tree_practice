# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Decision_tree_basics import Node, DecisionTree  # Ensure this module is available


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize your custom DecisionTree class
tree = DecisionTree(max_depth=5, min_samples_split=2)
print("Training successful")
# Fit the model
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)
print(predictions)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the custom DecisionTree: {accuracy:.2f}")