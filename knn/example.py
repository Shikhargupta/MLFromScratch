import random
from knn import *
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

random.seed(0)

data = load_wine()
X = data['data']
y = data['target']

N = X.shape[0]
ntrain = 140
ntest = N-ntrain

X, y = shuffle(X, y)

X_train, y_train = X[:ntrain], y[:ntrain]
X_test, y_test = X[ntrain:], y[ntrain:]

#Pre-process data
preprocessor = StandardScaler()
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)

#Predict using knn classifier
classifier = knn(k=5, isClassification=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = np.sum(y_pred==y_test).astype(float)/ntest
print("Accuracy: ", acc)


