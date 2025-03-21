from logistic_regression import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

X,y = make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,n_classes=2,random_state=1)

# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

def accuracy(y_pred,y_true):
    return np.sum(y_pred==y_true)

regressor = LogisticRegression(lr=0.01,num_epochs=1000)

regressor.fit(X,y)

y_pred = regressor.predict(X)

print(accuracy(y_pred,y)) # 97
