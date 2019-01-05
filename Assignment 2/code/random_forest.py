#Create a class RandomForest in ale called random_forest.py that takes in hyperparameters num_trees
#and max_depth andts num_trees random trees each with maximum depth max_depth. For prediction,
#have all trees predict and then take the mode.
from random_tree import RandomTree
import utils
import numpy as np

class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self,X,y):
        for i in range(self.num_trees):
            model = RandomTree(max_depth = self.max_depth)
            model.fit(X,y)
            self.trees.append(model)

    def predict(self,X):
        R,C = X.shape
        print(R)
        y_pred = []
        for i in range(self.num_trees):
            model = self.trees[i]
            y_pred.append(model.predict(X))


        y_pred_n = [[row[i] for row in y_pred] for i in range(X.shape[0])]
        binc = [np.bincount(row) for row in y_pred_n]
        pred = [np.argmax(row) for row in binc]

        return pred
