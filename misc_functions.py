'''
misc_functions.py
Collection of miscellaneous functions in Machine Learning
'''
import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def l2_distance(x,y):
    '''
    l2_distance
    Returns euclidean distance between 2 vectors

    Input
        x,y  : input vectors

    Output
        dist : euclidean distance between x and y
    '''
    dist = np.sqrt(np.sum((x-y)**2))
    return dist