import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2, kernel_poly, kernel_RBF
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")

        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        pl_kernel = kernelLogRegL2(kernel_fun=kernel_poly, lammy=0.01)
        pl_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(pl_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(pl_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(pl_kernel, Xtrain, ytrain)
        utils.savefig("PolyKernel.png")

        RBF_kernel = kernelLogRegL2(kernel_fun=kernel_RBF, lammy=0.01)
        RBF_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(RBF_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(RBF_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(RBF_kernel, Xtrain, ytrain)
        utils.savefig("RBFKernel.png")

    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # YOUR CODE HERE
        sigm= 10**np.arange(-2,3,dtype=float)
        lamm= 10**np.arange(-4,1,dtype=float)
        tr_err = np.inf
        val_err = np.inf
        tr_best = []
        val_best = []
        for sig in sigm:
            for lam in lamm:
                RBF_kernel = kernelLogRegL2(sigma=sig,kernel_fun=kernel_RBF, lammy=lam)
                RBF_kernel.fit(Xtrain, ytrain)
                loc_tr = np.mean(RBF_kernel.predict(Xtrain) != ytrain)
                loc_val = np.mean(RBF_kernel.predict(Xtest) != ytest)
                if loc_tr < tr_err:
                    tr_err = loc_tr
                    tr_best = [sig,lam]
                if loc_val < val_err:
                    val_err = loc_val
                    val_best = [sig, lam]
        print(tr_best)
        print(tr_err)
        print(val_best)
        print(val_err)

        RBF_kernel = kernelLogRegL2(sigma = tr_best[0] ,kernel_fun=kernel_RBF, lammy=tr_best[1])
        RBF_kernel.fit(Xtrain, ytrain)
        utils.plotClassifier(RBF_kernel, Xtrain, ytrain)
        utils.savefig("RBFbesttraining.png")

        RBF_kernel = kernelLogRegL2(sigma = val_best[0] ,kernel_fun=kernel_RBF, lammy=val_best[1])
        RBF_kernel.fit(Xtrain, ytrain)
        utils.plotClassifier(RBF_kernel, Xtrain, ytrain)
        utils.savefig("RBFbestvalidation.png")


    elif question == '4.1':
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(X.shape)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        print(Z.shape)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')

            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    else:
        print("Unknown question: %s" % question)
