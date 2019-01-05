import numpy as np
from utils import euclidean_dist_squared
import matplotlib.pyplot as plt
import os

class Kmeans:

    def __init__(self, k):
        self.k = k
        self.pred = []

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]
            errors = []
        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)


            # Update means
            for kk in range(self.k):
                if np.any(y==kk): # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y==kk].mean(axis=0)


            self.means = means
            pred = self.predict(X)
            err = self.error(X)
            print(err)      # FOR Q5.2


            changes = np.sum(y != y_old)

            if changes == 0:
                break


            # print('Running K-means, changes in cluster assignment = {}'.format(changes))
            # Stop if no point changed cluster

        self.means = means

    def predict(self, X):
        dist2 = euclidean_dist_squared(X, self.means)
        dist2[np.isnan(dist2)] = np.inf
        R = np.argmin(dist2, axis=1).shape
        self.pred = np.argmin(dist2,axis=1)
        return np.argmin(dist2, axis=1)

    def error(self, X):
        R,C = X.shape
        squared_dist = [(X[i] - self.means[self.pred[i]])**2 for i in range(R)]
        return np.sum(squared_dist)
