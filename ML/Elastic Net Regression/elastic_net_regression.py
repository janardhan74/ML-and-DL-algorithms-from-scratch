import numpy as np

class ElasticNetRegression():
    def __init__(self,alpha,l1_ratio,tol,num_epochs):
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        
    def soft_threshold(self,rho,lambda1):
        if rho < -lambda1:
            return rho + lambda1
        elif rho > lambda1:
            return rho - lambda1
        else:
            return 0
            
        
    def fit(self,X,y):
        self.bias = y.mean()
        y = y - y.mean()
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        
        lambda1 = self.alpha * self.l1_ratio
        lambda2 = self.alpha * (1-self.l1_ratio)
        
        for epoch in range(self.num_epochs):
            weights_old = self.weights.copy()

            for j in range(n_features):
                x_j = X[:,j]
                residual = y - (self.X @ self.weights) + self.weights[j] @ x_j
                rho = np.dot(x_j,residual)/n_samples
                self.weights = self.soft_threshold(rho,lambda1) (1+lambda2)
                
            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break
    def predict(self,X):
        return ( X @ self.weights) + self.bias
                
        