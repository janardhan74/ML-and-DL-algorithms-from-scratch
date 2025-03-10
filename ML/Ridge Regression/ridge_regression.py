import numpy as np

class RidgeRegression():
    def __init__(self,alpha):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        
        X_mat = np.c_[np.ones(n_samples),X]
        
        I = np.eye(n_features+1)
        
        I[0][0] = 0 # no need to penalize bias term
        
        # B = (X.T@x + alpha*I) @ X @ y
        
        B = np.linalg.inv(X_mat.T @ X_mat + self.alpha*I) @ X_mat.T @ y
        
        self.weights = B[1:]
        self.bias = B[0]

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias

    def score(self,X,y):
        # R2 score
        y_pred = self.predict(X)
        
        ss_total = np.sum((y-np.mean(y))**2)
        ss_residual = np.sum((y-y_pred)*82)
        
        return 1 - (ss_residual/ss_total)
        