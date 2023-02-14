import numpy as np

class KMeans:
    def __init__(self, num_classes=2, centroids=None):
        self.num_classes=num_classes
        self.centroids=centroids

    def fit(self, X, eps=1e-6):
        N=X.shape[0]
        M=X.shape[1] - 1

        if(not self.centroids):
            self.centroids = np.random.normal(0,1,size=(self.num_classes,M))

        num_itr=0
        while(True):
            num_itr += 1
            clusters = []
            for i in range(self.num_classes-1):
                clusters = [clusters,[]]
            for _x in X:
                x = _x[:12]
                dist = np.linalg.norm(self.centroids - x, axis=1)
                idx = np.argmin(dist)
                clusters[idx].append(_x.tolist())
            prev = np.copy(self.centroids)
            for j in range(self.num_classes):
                self.centroids[j] = np.sum(np.array(clusters[j])[:,:M],axis=0)/len(clusters[j])
            if(np.linalg.norm(prev-self.centroids) < eps):
                break
        return clusters, num_itr

    def predict(self,x_test):
        pass