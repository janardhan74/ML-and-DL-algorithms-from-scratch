import numpy as np

class LogisticRegression():
    def __init__(self,lr=0.001,num_epochs=1000):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        
    def _sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.num_epochs):
            
            z = np.dot(X,self.weights)+self.bias
            y_pred = self._sigmoid(z)
            
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            
            self.weights -= self.lr * dw            
            self.bias -= self.lr * db
            
            
    def predict(self,X):
        z = np.dot(X,self.weights)+self.bias
        y_pred = self._sigmoid(z)
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred
        