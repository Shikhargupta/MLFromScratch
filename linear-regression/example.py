import numpy as np
from LinearRegression import *
from sklearn import datasets
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

X, y = datasets.load_diabetes(return_X_y=True)
X, y = shuffle(X,y)

N=X.shape[0]
ntrain = 400
ntest = N-ntrain

X_train, y_train = X[:ntrain], y[:ntrain]
X_test, y_test = X[ntrain:], y[ntrain:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regressor = LinearRegression(lr=0.01, maxIter=1000)
regressor.fit_gradient_descent(X_train, y_train)
y_pred = regressor.predict(X_test)

# regressor = LinearRegression(lr=0.001, maxIter=1000)
# regressor.fit_closed_form(X_train, y_train)
# y_pred = regressor.predict(X_test)

error = np.sum((y_pred - y_test)**2)/ntest
print("Test Error: ", error)

plt.scatter(range(ntest), y_pred, label="predictions")
plt.scatter(range(ntest), y_test, label="ground truth")
plt.legend()
plt.title("Predictions vs Ground Truth")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# plt.plot(regressor.outs)
# plt.title("Loss Function vs Iterations")
# plt.xlabel("No. of iterations")
# plt.ylabel("Loss")
# plt.show()
