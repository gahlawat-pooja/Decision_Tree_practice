'''

This implementation is inspired by the research paper:
"Weighted Hybrid Decision Tree Model for Random Forest Classifier" by Vrushali Y. Kulkarni, Pradeep K. Sinha, and Manisha C. Petare.
The approach and methodology are referenced from the paper, but the implementation is written from scratch.

The Random Forest with Hybrid Decision Tree Model is a modification of the traditional Random Forest algorithm.
The model uses a hybrid split criterion for decision tree nodes, which randomly selects one of three measures: Gini Index, Entropy, or Gain Ratio.
The model also incorporates weighted voting based on Out-of-Bag (OOB) error to improve the accuracy of the ensemble predictions.

'''

import numpy as np
from collections import Counter
import gc
import random

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

# Random Forest with Hybrid Decision Tree Model
class RandomForest:
    def __init__(self, max_depth=3, min_samples_split=2, n_estimators=10, max_features=0.66, hybrid_split=True, weighted_voting=True):
        self.max_depth = max_depth                  
        self.min_samples_split = min_samples_split   
        self.n_estimators = n_estimators            
        self.max_features = max_features             
        self.hybrid_split = hybrid_split             # Enable hybrid split (gini, entropy, gain ratio).
        self.weighted_voting = weighted_voting       # Enable weighted voting based on OOB error.
        self.root_dict = {}                          # Dictionary to store each tree's root.
        self.oob_errors = {}                         # Dictionary to store Out-of-Bag (OOB) error for each tree.

    # Hybrid Split Selection (randomly chooses one of three measures)
    def _select_split_measure(self):
        return random.choice(["gini", "entropy", "gain_ratio"])

    # Calculate entropy
    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        ps = counts / len(s)
        entropy = 0
        for p in ps:
            if p > 0:
                entropy += p * np.log2(p)
        return -entropy 
    
    # Information Gain calculation
    def _information_gain(self, parent, left_child, right_child):
        parent_entropy = self._entropy(parent)
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        weighted_entropy = num_left * self._entropy(left_child) + num_right * self._entropy(right_child)
        return parent_entropy - weighted_entropy
       
    # Calculate Gini index
    @staticmethod
    def _gini_index(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        ps = counts / len(s)
        return 1 - np.sum(ps**2)

    # Calculate Gain Ratio
    def _gain_ratio(self, parent, left_child, right_child):
        gain = self._information_gain(parent, left_child, right_child)
        split_info = self._entropy(left_child) + self._entropy(right_child)
        return gain / (split_info + 1e-10)  # Avoid division by zero

    # Modified function to calculate Information Gain, Gini, or Gain Ratio dynamically
    def _calculate_split_gain(self, parent, left_child, right_child):
        split_measure = self._select_split_measure()
        if split_measure == "entropy":
            return self._information_gain(parent, left_child, right_child)
        elif split_measure == "gini":
            return (self._gini_index(parent) -
                    (len(left_child) / len(parent)) * self._gini_index(left_child) -
                    (len(right_child) / len(parent)) * self._gini_index(right_child))
        else:  # Gain Ratio
            return self._gain_ratio(parent, left_child, right_child)

    # Helper function to find the best split
    def _best_split(self, X, y):
        best_split = {}
        best_gain = -1
        n_rows, n_cols = X.shape  # Rows are samples, columns are features 
        max_repeat_split = 2
        for i_repeat_sub in range(max_repeat_split):
            if i_repeat_sub == 0:
                subsampled_cols = np.random.choice(n_cols, int(n_cols * self.max_features), replace=False)
            else:
                subsampled_cols = list(set(range(n_cols)) - set(subsampled_cols))
            for f_idx in subsampled_cols:
                X_curr = X[:, f_idx]
                # Try every possible split value for the feature
                for threshold in np.unique(X_curr):
                    # Concatenate features and target to form a combined dataset.
                    df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                    left_child = df[df[:, f_idx] <= threshold]
                    right_child = df[df[:, f_idx] > threshold]
                    gc.collect()  # Free up memory
                    
                    # Only consider splits that produce two non-empty groups
                    if len(left_child) > 0 and len(right_child) > 0:
                        y_left = left_child[:, -1]
                        y_right = right_child[:, -1]
                        
                        gain = self._calculate_split_gain(y, y_left, y_right)
                        
                        if gain > best_gain:
                            best_gain = gain
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

    # Recursive function to build the tree
    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        if n_rows >= self.min_samples_split and depth < self.max_depth:
            best = self._best_split(X, y)
            if best and best.get('gain', 0) > 0:
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth+1)
                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth+1)
                return Node(
                    feature=best['feature'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
            else:
                return Node(value=y)  # Create a leaf node storing target values.
        else:
            return Node(value=y)  # Base case: not enough samples or reached max depth.

    # Fit the random forest model to data
    def fit(self, X, y):
        print('n_estimators Total:', self.n_estimators)
        for estimator_i in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = self._build(X_sample, y_sample)
            self.root_dict[estimator_i] = tree
            # Compute OOB error for this tree using all samples in X.
            oob_predictions = np.array([self._predict(x, tree) for x in X])
            oob_error = np.mean(oob_predictions != y)
            self.oob_errors[estimator_i] = oob_error
            print(f'{estimator_i} estimator finished training, OOB Error: {oob_error:.4f}')

    # Recursive prediction on a single sample using a tree
    def _predict(self, x, tree):
        if tree.value is not None:
            # When reaching a leaf, return the majority class from the stored target values.
            return Counter(tree.value).most_common(1)[0][0]
        else:
            feature_value = x[tree.feature]
            if feature_value <= tree.threshold:
                return self._predict(x, tree.data_left)
            else:
                return self._predict(x, tree.data_right)

    # Predict using the ensemble of trees
    def predict(self, X):
        predictions = []
        for x in X:
            tree_votes = {}
            for estimator_i in range(self.n_estimators):
                pred = self._predict(x, self.root_dict[estimator_i])
                weight = 1 / (self.oob_errors[estimator_i] + 1e-6) if self.weighted_voting else 1
                pred = int(pred)
                tree_votes[pred] = tree_votes.get(pred, 0) + weight  
            predictions.append(max(tree_votes, key=tree_votes.get))
        return np.array(predictions)