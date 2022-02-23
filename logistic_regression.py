'''
Logistic Regression
Consists of classes/methods for binary and multiclass logistic
regression classification.
'''
import numpy as np
import random
from matplotlib import pyplot as plt
from cost_func import binary_cross_entropy_loss, grad_binary_cross_entropy_loss
from misc_functions import sigmoid

'''
BinaryLogisticReg
Class for binary logistic regression.
Methods :
    - init_w  : initialise weight vector
    - fit     : fit the model on training data
    - predict : predict labels for testing data
'''
class BinaryLogisticReg:
    def __init__(self):
        self.w = None
        random.seed(1)

    def init_w(self):
        limit = 1/np.sqrt(self.d)
        self.w = np.random.uniform(-limit,limit,self.d)

    def fit(self, X, y, MaxIter=10000, eta=0.01, eps=1e-6):
        self.eta = eta
        self.MaxIter = MaxIter

        self.d = X.shape[1]
        self.init_w()

        self.outs = []
        for i in range(MaxIter):
            p = sigmoid(X@self.w)
            self.w = self.w - eta*grad_binary_cross_entropy_loss(y,p,X)
            loss = binary_cross_entropy_loss(y, p)
            self.outs.append(loss)

            #Check convergence
            if i>1 and abs(self.outs[-1] - self.outs[-2])/abs(self.outs[-2]) < eps:
                break
    
    def predict(self,x):
        if self.w is None:
            print("Error: Model is not yet trained!")
        p = x@self.w > 0.5
        return p.astype(int)

