
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        # # YOUR CODE HERE FOR Q1.1.1

        # group_sum = ratings.groupby('item').sum()
        #
        # loc_max = group_sum['rating'].idxmax(axis=1)
        #
        # print(group_sum.loc['B000HCLLMM'])

        R,C = X.shape
        #print(R)
        #print(C)



        # YOUR CODE HERE FOR Q1.1.2

        group_sum = ratings.groupby('user').size()

        #
        # max = group_sum.idxmax(axis=1)
        #
        # print(group_sum.loc['A100WO06OQR8BQ'])
        #
        # print(max)

        # YOUR CODE HERE FOR Q1.1.3
        plt.figure()
        plt.yscale('log', nonposy='clip')
        ratings_per_user = X_binary.getnnz(axis=1)
        plt.hist(ratings_per_user)
        plt.title("ratings_per_user")
        plt.xlabel("number of users")
        plt.ylabel("number of ratings")
        filename = os.path.join("..", "figs", "ratings_per_user")
        plt.savefig(filename)

        plt.figure()
        plt.yscale('log',nonposy='clip')
        ratings_per_item = X_binary.getnnz(axis=0)
        plt.hist(ratings_per_item)
        plt.title("ratings_per_item")
        plt.xlabel("number of items")
        plt.ylabel("number of ratings")
        filename = os.path.join("..", "figs", "ratings_per_item")
        plt.savefig(filename)

        plt.figure()
        plt.hist(ratings['rating'])
        plt.title("ratings")
        plt.xlabel("number of ratings")
        plt.ylabel("rating")
        filename = os.path.join("..", "figs", "ratings")
        plt.savefig(filename)




    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        print("the index for grill brush is: " +str(grill_brush_ind))
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        model = NearestNeighbors(n_neighbors = 6, metric= 'euclidean')
        model.fit(X.T)
        euc_closest = model.kneighbors(grill_brush_vec.T,return_distance=False)
        print(euc_closest)

        model = NearestNeighbors(n_neighbors = 6,metric= 'euclidean')
        X_normal = normalize(X.T)
        model.fit(X_normal)
        normal_closest = model.kneighbors(grill_brush_vec.T,return_distance=False)
        print(normal_closest)

        model = NearestNeighbors(n_neighbors=6,metric='cosine')
        model.fit(X.T)
        cosine_closest = model.kneighbors(grill_brush_vec.T,return_distance=False)
        print(cosine_closest)

            # YOUR CODE HERE FOR Q1.3
        X_sums = X_binary.getnnz(axis=0)
        print("The # ratings of the items chosen by non-normal euclidean are:")
        print(X_sums[:][103866])
        print(X_sums[:][103865])
        print(X_sums[:][98897])
        print(X_sums[:][72226])
        print(X_sums[:][102810])

        print("The # ratings of the items chosen by cosine are:")
        print(X_sums[:][103866])
        print(X_sums[:][103867])
        print(X_sums[:][103865])
        print(X_sums[:][98068])
        print(X_sums[:][98066])

    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']
        R,C = X.shape

        # YOUR CODE HERE
        z = np.ones(500)
        z[400:500] = z[400:500] * .1
        d = np.diag(z)

        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,d)
        utils.test_and_plot(model,X,y,title="Least Squares with Weights", filename="weighted_least_squares.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")


    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE

        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_with_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares Poly with order" + str(p),filename="least_squares_poly_degree_"+str(p)+".pdf")

            # YOUR CODE HERE

    else:
        print("Unknown question: %s" % question)
