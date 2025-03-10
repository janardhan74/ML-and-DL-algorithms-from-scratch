from ridge_regression import RidgeRegression
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.linspace(0,10,100).reshape(-1,1)
y = 4 + 3*X.squeeze() + np.random.randn(100)

model = RidgeRegression(alpha=1.0)
model.fit(X, y)

y_pred = model.predict(X)


plt.scatter(X,y)
plt.plot(X,y_pred,c='red')
plt.show()
