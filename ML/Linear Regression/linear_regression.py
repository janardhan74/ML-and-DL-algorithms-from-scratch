import numpy as np

class LinearRegression():
    def __init__(self,lr = 0.01,num_epochs=100):
        self.num_epochs = num_epochs
        self.lr = lr
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        # y = mx + b
        
        for epoch in range(self.num_epochs):
            y_pred = np.dot(X,self.weights)
            # dw = 1/n * summation(1,n) 2x * (y_pred - y_true)
            # db = 1/n * summation(1,n) 2 * (y_pred - y_true)
            dw = 1/n_samples * np.dot( X.T , (y_pred - y))
            db = 1/n_samples * np.sum(y_pred-y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias