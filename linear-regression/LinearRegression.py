'''
LinearRegression.py
Implementation of Linear Regression algorithms 
from scratch.
'''

import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, maxIter=1000, eps=1e-5):
        self.lr = lr
        self.maxIter = maxIter
        self.eps = eps

    def fit_closed_form(self,X,y):
        X = np.insert(X,0,1,axis=1)
        self.w = np.linalg.inv(X.T@X)@X.T@y
    
    def loss(self,w):
        return 0.5*(np.linalg.norm(self.X@w - self.y))**2

    def grad(self,w):
        return self.X.T@(self.X@w - self.y)
        
    def fit_gradient_descent(self,X,y):
        X = np.insert(X, 0, 1, axis=1)
        self.X = X
        self.y = y

        m = X.shape[0]
        n = X.shape[1]
        self.w = np.random.rand(n)
        self.outs = [self.loss(self.w)]

        for i in range(self.maxIter):
            self.w = self.w - self.lr*self.grad(self.w)/m
            self.outs.append(self.loss(self.w))

            if(abs(self.outs[-2] - self.outs[-1])/abs(self.outs[-2]) < self.eps):
                break

    def predict(self,X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        return X_test@self.w
        













