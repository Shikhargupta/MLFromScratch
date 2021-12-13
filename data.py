'''
data.py
Create data for machine learning algorithms
'''
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white')

'''
create_lin_reg_data
Generates data points for the linear regression problem.

Input arguments:
    n: the degree of the polynomial (default 1)
    addNoise: if true, add gaussian noise to the data

TODO: 
    Add support for polynomial data
    Allow tuning the parameters of noise
'''
def create_lin_reg_data(n=1, addNoise=True):
    a = random.uniform(1,10)
    b = random.uniform(1,10)

    x = np.arange(1,10,0.1)
    y = list(map(lambda xs: a*xs + b, x))

    if(addNoise):
        noise = np.random.normal(0,2,len(y))
        y = y + noise

    return x,y,a,b

def plot_data():
    x,y,a,b = create_lin_reg_data()

    y_true = list(map(lambda g: a*g+b, x))
    plt.plot(x,y_true, label="True Line", color='r')
    sns.scatterplot(x=x,y=y, label="Noisy Data")
    plt.title("Linear Regression Data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



