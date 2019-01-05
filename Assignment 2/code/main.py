# basics
import os
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# our code
import utils
from knn import KNN
from naive_bayes import NaiveBayes
from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
    #    print("Training error: %.3f" % tr_error)
    #    print("Testing error: %.3f" % te_error)


    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        test_errors = []
        training_errors = []
        k_values = []

        for k in range(1,16):
            model = DecisionTreeClassifier(max_depth=k, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)

            test_errors.append(te_error)
            training_errors.append(tr_error)
            k_values.append(k)

#        print(k_values)
#        print(test_errors)
#        print(training_errors)
        plt.plot(k_values,test_errors,'r--', label="test_errors")
        plt.plot(k_values,training_errors,'bs', label="training errors")
        plt.legend()
        plt.xlabel("Depth")
        plt.ylabel("Error")
        fname = os.path.join("..", "figs", "q1.1_errors.pdf")
        plt.savefig(fname)

        #    print("N = " + str(k) + " " + "Training error: %.3f" % tr_error)
        #    print("N = " + str(k) + " " + "Testing error: %.3f" % te_error)




    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        # print(len(X), len(y), len(X_test), len(y_test)) ## for slicing
        # training_set_x = X[0:200]
        # validation_set_x = X[200:400]
        # training_set_y = y[0:200]
        # validation_set_y = y[200:400]
        ## switching training_set and validation_set
        training_set_x = X[200:400]
        validation_set_x = X[0:200]
        training_set_y = y[200:400]
        validation_set_y = y[0:200]
        # X_test, y_test = dataset["Xtest"], dataset["ytest"]
        for k in range(1,16):
            model = DecisionTreeClassifier(max_depth=k, criterion='entropy', random_state=1)
            model.fit(training_set_x, training_set_y)

            y_pred = model.predict(training_set_x)
            tr_error = np.mean(y_pred != training_set_y)

            y_pred = model.predict(validation_set_x)
            te_error = np.mean(y_pred != validation_set_y)
            print("N = " + str(k) + " " + "Training error: %.3f" % tr_error)
            print("N = " + str(k) + " " + "Testing error: %.3f" % te_error)


    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        #print(wordlist[50]) #for q2.2.1
        #print(X[501])
        # i = 0     #for q.2.2.2
        # for a in X[501]:
        #     i = i + 1
        #     if a == 1:
        #         print(wordlist[i])
        #print(groupnames[y[500]]) #for q.2.2.3


    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model2 = BernoulliNB()
        model2.fit(X,y)
        y_pred = model2.predict(X_valid)
        v_error2 = np.mean(y_pred != y_valid)
        print("Scikit learn bernouilli nb validation error: %.3f" % v_error2)

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        for k in [1,3,10]:
            model = KNN(k)
            model.fit(X,y)
            y_pred = model.predict(Xtest)
            v_error = np.mean(y_pred != ytest)
            print("KNN test error: %.3f" % v_error)
            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
            print("N = " + str(k) + " " + "Training error: %.3f" % tr_error)
            if k == 1:
                utils.plotClassifier(model, X, y)
                fname = os.path.join("..", "figs", "q3_KNN_plot.pdf")
                plt.xlabel("x-coordinate")
                plt.ylabel("y-coordinate")
                plt.legend()
                plt.savefig(fname)
                print("KNN Plot is saved")
                model = KNeighborsClassifier(n_neighbors = k)
                model.fit(X,y)
                utils.plotClassifier(model, X, y)
                fname = os.path.join("..", "figs", "q3_scikit_KNeighbors_plot.pdf")
                plt.xlabel("x-coordinate")
                plt.ylabel("y-coordinate")
                plt.legend()
                plt.savefig(fname)
                print("Scikit_Kneighbors plot is saved")



    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        # print("Decision tree info gain")
        # evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        # print("Random tree  with max depth = inf")
        evaluate_model(RandomTree(max_depth=np.inf))
        evaluate_model(DecisionTree(max_depth=np.inf))
        evaluate_model(RandomForest(num_trees = 50,max_depth =np.inf))
        evaluate_model(RandomForestClassifier(max_depth=None,n_estimators=50))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']
        R,C = X.shape
        #X is 500x2
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)

        # print(model.error(X))
        # R = y.shape

        #y is 500x1
        # print(err)

        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        errors = []
        model = Kmeans(k=4)
        for i in range(50):
            model.fit(X)
            y= model.predict(X)
            err= model.error(X)
            errors.append([err, y])

        lst_sorted = sorted(errors, key=lambda tup: tup[0])
        y_pred = lst_sorted[0][1]
        plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="jet")
        fname = os.path.join("..", "figs", "kmeans_least_error.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        errors = []
        min_k = []
        m = 0
        for k in range(1,11):
            for i in range(50):
                model = Kmeans(k)
                model.fit(X)
                y= model.predict(X)
                err= model.error(X)
                errors.append([err, k])
            lst_sorted = sorted(errors, key=lambda tup: tup[0])
            min_k.append(lst_sorted[0])
        get_errors = [item[0] for item in min_k]
        get_ks = [item[1] for item in min_k]
        plt.plot(get_ks,get_errors,'r--')
        fname = os.path.join("..", "figs", "kmeans_varying_k_error.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']
        model = DBSCAN(eps=18, min_samples=100)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)

    else:
        print("Unknown question: %s" % question)
