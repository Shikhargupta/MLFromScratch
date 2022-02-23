import random
import pandas as pd
from logistic_regression import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

################ Logistic Regression ########################
random.seed(1)

#Load data
data = pd.read_csv("logistic_reg_data.csv")
X = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

N = X.shape[0]
ntrain = int(N*0.8)
ntest = N-ntrain

#Pre processing
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Add bias feature
X = np.c_[X, np.ones(N)]

#Split data into train and test
X_train, X_test = X[:ntrain,:], X[ntrain:, :]
y_train, y_test = y[:ntrain], y[ntrain:]

#Run model and get predictions
model = BinaryLogisticReg()
model.fit(X_train, y_train, eta=0.0005)
y_pred = model.predict(X_test)

#Get accuracy
acc = np.sum((y_pred==y_test).astype(float))/ntest
print("Accuracy on test data:", acc)

plt.plot(model.outs)
plt.title("Convergence Plot")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

#Plots
x_class0 = X[y==0]
x_class1 = X[y==1]

line_x = np.linspace(-2.0,2,100)
line_y = (model.w[0]*line_x + model.w[2])/(-model.w[1])

plt.scatter(x_class0[:,0], x_class0[:,1], label="class A")
plt.scatter(x_class1[:,0], x_class1[:,1], label="class B")
plt.plot(line_x, line_y, color='r', label="decision line")
plt.legend()
plt.title("Visualization of Classification by Logistic Regression")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()