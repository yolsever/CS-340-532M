import numpy as np
from decision_tree import DecisionTree

class SimpleDecision(DecisionTree):

    def predict(self, X):
        splitmodelsplitVar = self.splitModel.splitVariable
        splitmodelsplitVal = self.splitModel.splitValue
        splitmodelsplitSat = self.splitModel.splitSat
        splitmodelsplitNot = self.splitModel.splitNot
        submodel0splitVar =  self.subModel0.splitModel.splitVariable
        submodel0splitVal = self.subModel0.splitModel.splitValue
        submodel0splitSat = self.subModel0.splitModel.splitSat
        submodel0splitNot = self.subModel0.splitModel.splitNot
        submodel1splitVar = self.subModel1.splitModel.splitVariable
        submodel1splitVal = self.subModel1.splitModel.splitValue
        submodel1splitSat = self.subModel1.splitModel.splitSat
        submodel1splitNot = self.subModel1.splitModel.splitNot

        M, D = X.shape
        y = np.zeros(M)

        for n in range(M):
            if X[n,splitmodelsplitVar] > splitmodelsplitVal:
                if X[n,submodel1splitVar] > submodel1splitVal:
                    y[n] = submodel1splitSat
                else:
                    y[n] = submodel1splitNot
            else:
                if X[n,submodel0splitVar] > submodel0splitVal:
                    y[n] = submodel0splitSat
                else:
                    y[n] = submodel0splitNot
        return y
