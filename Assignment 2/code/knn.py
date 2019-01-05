"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from statistics import mode
import utils
from collections import Counter

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y
#        print(X)
#        print(y)

    def predict(self, Xtest):
        edist = []
        pred = []

        for x in Xtest:
            edist = [((x[0]-self.X[i][0])**2+(x[1]-self.X[i][1])**2)**(1/2) for i in range(len(self.X[:]))]
            #print(edist)
            temp = []
            for c in range(len(self.X[:])):
                temp.append([edist[c],c])

            x_sorted = sorted(temp,key=lambda sl: (sl[0],sl[1]))

            x_neighbors = [x_sorted[i][1] for i in range(self.k)]

            nearest_labels = self.y[x_neighbors]

            binc = np.bincount(nearest_labels)
            mod = np.argmax(binc)

            pred.append(mod)

            return pred
        # edist = utils.euclidean_dist_squared(Xtest,self.X)
        #
        # R,C = edist.shape
        # TR, TC = Xtest.shape

        # edist_sort = np.argsort(np.argsort(edist))
        # print(edist_sort)
        #
        # neighbours = np.zeros((TR,self.k))
        # y_pred = np.zeros(TR)
        # x = []

        # for  r in range(R):
        #     for c in range(C):
        #         x.append([edist[r][c],(r,c)])
        #
        # x_sorted = sorted(x,key=lambda sl: (sl[0],sl[1]))

        # for r in range(R):
        #     for c in range(C):
        #         x = edist_sort[r][c]
        #         if x < self.k:
        #             neighbours[r][x] = c        ### I THINK THE ERROR IS HERE
        #
        # neighbours = neighbours.astype(np.int64)
        #
        # #print(neighbours)
        #
        # for r in range(TR):
        #     for c in range(self.k):
        #         neighbours[r][c] = self.y[neighbours[r][c]]
        #
        # #print(neighbours)
        #
        # mod = stats.mode(neighbours,axis=1)
        # y_pred = mod[0]
        # # y_pred_flattened = y_pred.flatten()
        # # print(np.bincount(y_pred_flattened))
        #
        # return y_pred
