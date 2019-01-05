import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from neural_net import NeuralNet
from manifold import MDS, ISOMAP
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape
        print("n =", n)
        print("d =", d)

        f1, f2 = np.random.choice(d, size=2, replace=False)

        plt.figure()
        plt.scatter(X[:,f1], X[:,f2])
        plt.xlabel("$x_{%d}$" % f1)
        plt.ylabel("$x_{%d}$" % f2)
        for i in range(n):
            plt.annotate(animals[i], (X[i,f1], X[i,f2]))

        utils.savefig('two_random_features.png')

    elif question == "1.1":
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        print(X.shape[0])
        animals = dataset['animals']
        n,d = X.shape
        print("n =", n)
        print("d =", d)

        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        plt.figure()
        plt.scatter(X[:,0], X[:,1])

#        plt.xlabel("$x_{%d}$ component 1")
#        plt.ylabel("$x_{%d}$ component 2")
        print(np.sum(pca.explained_variance_ratio_))
        for i in range(n):
            plt.annotate(animals[i], (X[i,0], X[i,1 ]))

        utils.savefig('PCA_2_features.png')

    elif question == "1.2":
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        pca = PCA(n_components=5)
        X = pca.fit_transform(X)
        print(np.sum(pca.explained_variance_ratio_)) # for 2 it is 0.32313181181376877

        ## with 5 components we have 0.5446592680228373 variance explained

    elif question == '1.3':

        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('MDS_animals.png')

        pca = PCA(n_components=2)
        pca.fit(X)
        Z = pca.transform(X)

        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        f = 0.0
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)
        print(f)


        ## error with MDS solution is 1776.8183112784675
        ## error with PCA solution is 4942.195453022984

    elif question == '1.4':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        for n_neighbours in [2,3]:
            model = ISOMAP(n_components=2, n_neighbours=n_neighbours)
            Z = model.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z[:,0], Z[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(animals[i], (Z[i,0], Z[i,1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)

    elif question == '1.5':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        X =  TSNE(n_components=2).fit_transform(X)
        plt.figure()
        plt.scatter(X[:,0], X[:,1])

        for i in range(n):
            plt.annotate(animals[i], (X[i,0], X[i,1 ]))

        utils.savefig('TSNE_2_features.png')


    elif question == "2":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50]
        model = NeuralNet(hidden_layer_sizes)

        t = time.time()
        model.fit(X,Y)
        print("Fitting took %d seconds" % (time.time()-t))

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == "2.4":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        print("n =", X.shape[0])
        print("d =", X.shape[1])

        model = MLPClassifier(tol=1e-3,learning_rate='invscaling',max_iter= 10)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)
