from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from linear_regression import LinearRegression

RANDOM_STATE = 42
X,y = make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=7,random_state=42)


plt.scatter(X,y)
# plt.show()

def mean_squared_error(y_true,y_pred):
    return np.sum((y_true-y_pred)**2)/y_true.shape[0]

regressor = LinearRegression(num_epochs=500)
regressor.fit(X,y)

y_pred = regressor.predict(X)

print(mean_squared_error(y,y_pred))
plt.plot(X,y_pred,c='red')
plt.show()

