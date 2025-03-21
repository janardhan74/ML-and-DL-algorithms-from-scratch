from decision_tree_classification import DecisionTreeClassifier
import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self , n_trees=10 , max_depth=10 , min_sample_split=2 , n_features=None ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = None
        
    def fit(self,X,y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          n_features=self.n_features,
                                          min_sample_split=self.min_sample_split)

            X_samples,y_samples = self._bootstrap_samples(X,y)
            tree.fit(X_samples,y_samples)
            self.trees.append(tree)

    def _bootstrap_samples(self,X,y):
        n_samples = X.shape[1]
    
        idxs = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def _most_common_label(self,y):
        ctr = Counter(y)
        most_common = ctr.most_common(1)[0][0]
        return most_common

    def predict(self,X):
        tree_preds = [tree.predict(X) for tree in self.trees]
        
        tree_preds = np.swapaxes(tree_preds,0,1)

        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        
        return predictions
        
        
        
        