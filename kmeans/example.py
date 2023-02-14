import numpy as np
from k_means import *
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("/Users/shikhar/Desktop/MLFromScratch/kmeans/winedata.csv")
X = data.to_numpy()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.insert(X,X.shape[1],0,axis=1)
X[150:, -1] = 1


model = KMeans()
clusters, num_itr = model.fit(X)

print("done")