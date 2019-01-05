import numpy as np
from numpy.linalg import norm
import utils
from sklearn.decomposition import PCA
from findMin import findMin

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        pca = PCA(k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z, f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        print(f)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)
                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()



class ISOMAP(MDS):

    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.nn = n_neighbours

    def compress(self, X):
        n = X.shape[0]

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        assert np.allclose(D,D.T)

        # TODO: Convert these Euclidean distances into geodesic distances
        D_args= np.argsort(D)
        D_indx = np.argsort(D_args)

        for i in range(n):
            for j in range(n):
                if D_indx[i,j] <= self.nn:
                    D_indx[j,i] = D_indx[i,j]

        adj = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if D_indx[i,j] <= self.nn:
                    adj[i,j] = 1
                if i == j:
                    adj[i,j] = 0

        assert np.allclose(adj,adj.T)

        for i in range(n):
            for j in range(n):
                D[i,j] = utils.dijkstra(adj,i,j)

        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.
        D[np.isinf(D)] = D[~np.isinf(D)].max()

        # Initialize low-dimensional representation with PCA
        pca = PCA(self.k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z,f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, self.k)
        return Z
