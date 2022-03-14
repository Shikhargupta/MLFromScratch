import sys
sys.path.append('../')

import numpy as np
from scipy import stats
from misc_functions import l2_distance

class knn:
    '''
    knn
    Implementation of k-nearest neighbors algorithm.
    Methods :
        - fit     : fit the model on training data
        - predict : predict labels/values for testing data
    '''
    def __init__(self, k, isClassification=True):
        self.k = k
        self.classify = isClassification

    def fit(self,X,y):
        #Store training samples
        self.X = X
        self.y = y

    def predict(self,x):
        y_pred = []
        for x_test in x:
            distances = []
            for _x in self.X:
                #Calculate euclidean distance from each training sample
                distances.append(l2_distance(_x,x_test))

            #Get k nearest neighbors
            idx = np.argsort(np.array(distances))
            k_neighbors = self.y[idx[:self.k]]

            if self.classify:
                #Get class label with highest frequency
                y_pred.append(stats.mode(k_neighbors)[0][0])
            else:
                #Take average
                y_pred.append(np.mean(k_neighbors))

        return np.array(y_pred).astype(int)
