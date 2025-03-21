import numpy as np
from collections import Counter

class Node():
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature,
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        
    def is_leaf_node(self): 
        return self.value is not None


class DecisionTreeClassifier():
    def __init__(self,max_depth=2,n_features=None,min_sample_split=100):
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_sample_split = min_sample_split
        self.root = None
        
    def fit(self,X,y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features,X.shape[1])
        
        self.root = self._grow_tree(X,y)
        
    def _grow_tree(self,X,y,depth=0):
        
        n_labels = len(np.unique(y))
        n_samples,n_features = X.shape

        # if we met stopping criteria
        if n_labels==1 or depth >= self.max_depth or n_samples < self.min_sample_split:
            most_common_label = self._most_common_label(y)
            return Node(value=most_common_label)
        
        # get random n features
        feat_idxs = np.random.choice(n_features,self.n_features,replace=False) # don't allow duplicates

        # split based on best split
        best_feature,best_threshold = self._best_split(X,y,feat_idxs)
        left_idxs,right_idxs = self._split(X[:,best_feature],best_threshold)

        left = self._grow_tree(X[left_idxs,:],y,depth+1)
        right = self._grow_tree(X[right_idxs,:],y,depth+1)

        return Node(feature=best_feature,threshold=best_feature,left=left,right=right)


    def _best_split(self,X,y,feat_idxs):
        best_gain = -1
        
        best_feature,best_threshold = None,None
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                gain = self._gain(X_column,y,feat_idx,thr)
                
                if gain>best_gain:
                    best_gain=gain
                    best_feature=feat_idx
                    best_threshold=thr
                    
        return best_feature,best_threshold
        
            
            
    def _gain(self,X_column,y,feature,threshold):
        # parent entropy - child entropy
        parent_entropy = self._entropy(y)
        
        # split based on feat and thr

        left_idxs , right_idxs = self._split(X_column,threshold)
        
        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        # print("over")
        # print(X_column)
        e_l,e_r = self._entropy(y[left_idxs]),self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        
        return parent_entropy-child_entropy

        
    def _split(self,X_column,threshold):
        left_idxs = np.argwhere(X_column<=threshold).flatten()
        right_idxs = np.argwhere(X_column>threshold).flatten()
        # output look like that with out faltten
        #     array([[2],
        #    [3],
        #    [7],
        #    [8]], dtype=int64)
        return left_idxs,right_idxs
        
        
        
    def _entropy(self,y):
        # print("hi")
        # print(y)
        # print("hi")
        hist = np.bincount(y)
        n = len(y)
        px = hist/n
        return -np.sum([p*np.log(p) for p in px if p > 0])
            
    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
        
    def predict(self,X):
        preds = [self._traverse(sample,self.root) for sample in X]
        
        return preds
    
    def _traverse(self,X,root):
        if root.is_leaf_node():
            return root.value
        
        if X[root.feature]<=root.threshold :
            return self._traverse(X,root.left)
        else:
            return self._traverse(X,root.right)
        