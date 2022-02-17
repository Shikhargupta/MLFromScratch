from logistic_regression import *
import pandas as pd
from sklearn.preprocessing import StandardScaler

################ Logistic Regression ########################

data = pd.read_csv("logistic_reg_data.csv")
X = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


