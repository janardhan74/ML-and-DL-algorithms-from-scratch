import numpy as np
    

# lasso regression uses cordinate descent instead of gradient descent due |W|
# the loss fun formula is L = summation((y-y_pred)**2) + alpha * |w|

# Note
# when the values are not scaled -> the weights for the feature having less scale are shirnked first & dominated -> becoz of denominator ||x_j||**2 

class LassoRegression():
    def __init__(self,num_epochs=1000,alpha=1.0,tol=1e-4):
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.tol = tol
        
        self.weights = None
        self.bias = None
        
        
    def soft_threshold(self,rho,alpha):
        if rho < - alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            return 0
        
        
    def fit(self,X,y):
        n_samples,n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = y.mean()
        
        y = y - y.mean()

        for _ in range(self.num_epochs):
            weights_old = self.weights.copy()
            
            for j in range(n_features):
                x_j = X[:,j]
                residual =  y - (X @ self.weights) + (self.weights[j] * x_j)
                rho = np.dot(x_j,residual)/n_samples
                # summation(i:1,n) x_ij * (y_i - summation(k!=j) z_ik * w_k) # actual formula 
                self.weights[j] = self.soft_threshold(rho,self.alpha)
                # soft_threshold(rho,alpha) / ||x_j||**2 when the values are not scaled 
                
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break
        
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
        
    