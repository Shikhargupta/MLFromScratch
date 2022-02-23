'''
cost_func
Collection of various cost functions and their gradients 
used in Machine Learning
'''

import numpy as np

def least_squares(y_true, y_pred):
    return (y_true**2 - y_pred**2)**0.5



def binary_cross_entropy_loss(y_true, p):
    '''
    binary_cross_entropy_loss
    Returns total cross entropy loss for binary classes

    Inputs
        y_true : ground truth labels
        p      : probabilities

    Outputs
        loss : total loss over the training batch
    '''
    loss = -np.sum(y_true*np.log(p) + (1-y_true)*np.log(1-p))/y_true.shape[0]
    return loss



def grad_binary_cross_entropy_loss(y_true, p, X):
    '''
    grad_binary_cross_entropy_loss
    Returns total gradient of cross entropy loss for 
    binary classes

    Inputs
        y_true : ground truth labels
        p      : probabilities
        X      : training data

    Outputs
        grad : total gradient over the training batch
    '''
    grad = np.sum(X*(p - y_true).reshape(p.shape[0],1))
    return grad