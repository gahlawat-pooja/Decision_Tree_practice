'''


This code tries to recreate code from following paper: 
    Jain, Vikas, Ashish Phophalia, and Jignesh S. Bhatt. 
    "Investigation of a joint splitting criteria for decision tree classifier use of information gain and gini index." 
    In TENCON 2018-2018 IEEE Region 10 Conference, pp. 2187-2192. IEEE, 2018.
     
'''    
import numpy as np
from collections import Counter
import gc

# Node class to represent each node of the tree
class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value
    

# Random Forest class
class RandomForest:
    def __init__(self, max_depth= 3, min_samples_split=2, n_estimators=10, max_features = 0.66, alpha=0.5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.alpha = alpha  # New parameter for joint criteria
        self.root_dict = {}

    # Helper function to calculate entropy  
    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        ps = counts / len(s)
    
        # Calculate entropy
        entropy= 0
        for p in ps:
            if p > 0:
                entropy = entropy + p * np.log2(p)
        return -entropy 

    # Helper function to calculate Gini Index
    def _gini_index(self, y):
        counts = np.bincount(np.array(y, dtype= np.int64))
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    # Helper function to calculate Information Gain
    def _information_gain(self, parent, left_child, right_child):
        parent_entropy = self._entropy(parent)
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        weighted_entropy = num_left * self._entropy(left_child) + num_right * self._entropy(right_child)
        return parent_entropy - weighted_entropy

    # Helper function to find the best split using joint criteria
    def _best_split(self, X, y):
        best_split = {}
        best_gain = -1
        n_rows, n_cols = X.shape  # rows are samples, cols are features 
        k = int(np.sqrt(n_cols))  # Select k = sqrt(M) attributes
        max_repeat_split = 2
        
        for i_repeat_sub in range(max_repeat_split):
            if i_repeat_sub == 0:
                subsampled_cols = np.random.choice(n_cols, k, replace=False)
            else:
                subsampled_cols = list(set(range(n_cols)) - set(subsampled_cols))  # for every dataset feature
            
            for f_idx in subsampled_cols:
                X_curr = X[:, f_idx]
                # for every possible split
                for threshold in np.unique(X_curr):
                    df = np.concatenate((X, y.reshape(1, -1).T), axis=1) #axis=1 means concatenation happens along columns
                    left_child = df[df[:, f_idx] <= threshold]
                    right_child = df[df[:, f_idx] > threshold]
                    gc.collect()

                    # do calculation only if there's data in both subsets
                    if len(left_child) > 0 and len(right_child) > 0:
                        y = df[:, -1]
                        y_left = left_child[:, -1]
                        y_right = right_child[:, -1]
                        # Calculate Information Gain and Gini Index
                        gain = self._information_gain(y, y_left, y_right)
                        gini = self._gini_index(y)
                        joint_value = self.alpha * gain + (1 - self.alpha) * (1 - gini)

                        # Update best split if joint value is higher than current best
                        if joint_value > best_gain:
                            best_gain = joint_value
                            best_split = {
                                'feature': f_idx,
                                'threshold': threshold,
                                'df_left': left_child,
                                'df_right': right_child,
                                'gain': gain
                            }
            if best_split:
                break
        return best_split

    # Stopping criteria for splitting
    def stop_split(self, D, depth):
        nmin = self.min_samples_split
        if len(D) < nmin:
            return True
        elif np.all(np.unique(D[:, :-1], axis=0).shape[0] == 1):  # Check if all attributes are constant
            return True
        elif depth >= self.max_depth:  # Check if max depth has been reached
            return True
        elif np.all(D[:, -1] == D[0, -1]):  # Check if all outputs belong to the same class
            return True
        else:
            return False

    # Recursive function to build the tree
    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        # Check if a node should be a leaf node
        if self.stop_split(X, depth):
            return Node(value=y)

        best = self._best_split(X, y)
        # If best split is not pure
        if best and best['gain'] > 0:
            # Create left and right child
            left = self._build(
                X=best['df_left'][:, :-1],
                y=best['df_left'][:, -1],
                depth=depth + 1
            )
            right = self._build(
                X=best['df_right'][:, :-1],
                y=best['df_right'][:, -1],
                depth=depth + 1
            )
            # Return the node with left and right child
            return Node(
                feature=best['feature'],
                threshold=best['threshold'],
                data_left=left,
                data_right=right,
                gain=best['gain']
            )
        else:       
            return Node(value=y)

    def fit(self, X, y):
        # Call recursive function to build the tree
        print('n_estimators Total : ', self.n_estimators)  # print total no. of trees 
        for estimator_i in range(self.n_estimators):  # loop that will iterate through the specified number of trees
            # Sample X and y with replacement (bootstrap sampling)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            assert X.shape[0] == X_sample.shape[0]
            self.root_dict[estimator_i] = self._build(X_sample, y_sample)  # This dictionary is used to store the root nodes of all the trees in the forest
            print(f'{estimator_i} estimator finished training')

    def _predict(self, x, tree):
        # Leaf node
        if tree.value is not None:
            return tree.value  # If the node is a leaf, return all labels in this leaf directly
        else:
            feature_value = x[tree.feature]  
            # Go to the left
            if feature_value <= tree.threshold:
                return self._predict(x=x, tree=tree.data_left)
            else:
                # Go to the right
                if feature_value > tree.threshold:
                    return self._predict(x=x, tree=tree.data_right)  

    # This function predicts class labels for single data point 'x' using ensemble
    def _ensemble_predict(self, x, ensemble_dict):
        tree_predictions = []
        for estimator_i in range(self.n_estimators):
            tree_predictions.append(self._predict(x=x, tree=ensemble_dict[estimator_i]))
        # Concatenated all as one numpy vector
        tree_predictions = np.concatenate(tree_predictions)
        return Counter(tree_predictions).most_common(1)[0][0]

    def predict(self, X):
        # Predict the class label for each data point in 'X' using the ensemble 
        return [self._ensemble_predict(x, self.root_dict) for x in X]
