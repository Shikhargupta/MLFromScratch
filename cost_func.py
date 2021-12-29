'''
cost_func
Database for definitions of various cost functions
used in Machine Learning
'''

def least_squares(y_true, y_pred):
    return (y_true**2 - y_pred**2)**0.5