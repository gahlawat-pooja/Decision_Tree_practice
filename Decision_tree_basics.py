#This file is recreated using the concept and code of decision tree from "Better Data Science" by  Dario Radečić.
# Visit following website "Better Data Science" for original content and more details (https://betterdatascience.com/mml-decision-trees/)
# row 117 to 120 of _build func (last rows) have 2 extra else statements than original code and also _predict function has two extra else statements just for understanding
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
    

# Decision Tree class
class DecisionTree:
    def __init__(self, max_depth= 3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    # Helper function to calculate entropy  
    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        ps = counts / len(s)
    
    #calculate entropy
        entropy= 0
        for p in ps:
            if p > 0:
                entropy = entropy + p * np.log2(p)
        return -entropy 
     
    # Helper function to calculate Information Gain
    def _information_gain(self, parent, left_child, right_child):
        parent_entropy = self._entropy(parent)
        num_left= len(left_child)/ len(parent)
        num_right = len(right_child)/ len(parent)
        weighted_entropy = num_left * self._entropy(left_child) + num_right * self._entropy(right_child)
        return parent_entropy - weighted_entropy

    # Helper function to find the best split
    def _best_split(self, X, y):
        best_split = {}
        best_gain = -1
        n_rows, n_cols = X.shape  #rows are samples, cols are features 
    
        #for every dataset feature
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            #for every possible split
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1) #axis=1 means concatenation happens along columns
                left_child = df[df[:, f_idx] <= threshold]
                right_child = df[df[:, f_idx] > threshold]
                gc.collect()
                
                
                #do calculation only if there's data in both subset
                if len(left_child) > 0 and len(right_child) > 0:
                    
                    y = df[:, -1]
                    y_left = left_child[:, -1]
                    y_right = right_child[:, -1]
                    #calculate information gain
                    gain = self._information_gain(y, y_left, y_right)
                    #update best split if gain is higher than current best
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            'feature': f_idx,
                            'threshold': threshold,
                            'df_left': left_child,
                            'df_right': right_child,
                            'gain': gain
                            }
        return best_split
    # Recursive function to build the tree
    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        #check if a node should be a leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            print(X.shape, y.shape)
            #if yes, calculate best split
            best = self._best_split(X, y)
            #if best split is not pure
            if best and best['gain'] > 0:
                #create left and right child
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth+1
                    )
                print('Left Finished')
                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth+1
                    )
                print('Right Finished')
                #return the node with left and right child
                return Node(
                    feature=best['feature'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                    )
            else:       
                return Node(value= Counter(y).most_common(1)[0][0])
        else:       
            #if best split is pure, return leaf node
            return Node(value= Counter(y).most_common(1)[0][0])
    def fit(self, X, y):
        #call recursive function to build the tree
        self.root = self._build(X, y)
    
    def _predict(self, x, tree):
    
        #leaf node
        if tree.value != None:
            return tree.value
        else:
            feature_value = x[tree.feature]  
            #Go to the left
            if feature_value <= tree.threshold:
                return self._predict(x=x, tree = tree.data_left)
            else:
            #Go to the right
                if feature_value > tree.threshold:
                    return self._predict(x=x, tree = tree.data_right)  
        
    
    def predict(self, X):
        #call the _predict function
        return [self._predict(x, self.root) for x in X]   
                    




        
        