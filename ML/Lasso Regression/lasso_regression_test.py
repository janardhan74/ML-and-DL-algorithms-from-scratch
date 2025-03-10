from lasso_regression import LassoRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.linspace(0,10,100).reshape(-1,1)
y = 4 + 3*X.squeeze() + np.random.randn(100)

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale before training

model = LassoRegression(alpha=0.1, num_epochs=100000, tol=1)
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c='red')
plt.show()



plt.scatter(X,y)
plt.plot(X,y_pred,c='red')
plt.show()
