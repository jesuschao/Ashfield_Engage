from k_means_constrained import KMeansConstrained
import warnings
import base
from scipy.spatial.distance import cdist
class SameSizeKMeansMinCostFlow(base.Base):

    def __init__(self, n_clusters, max_iters=1000, distance_func=cdist, random_state=42):
        '''
        Args:
            n_clusters (int): number of clusters
            max_iters (int): maximum iterations
            distance_func (object): callable function with input (X, centers) / None, by default is l2-distance
            random_state (int): random state to initiate, by default it is 42
        '''
        super(SameSizeKMeansMinCostFlow, self).__init__(n_clusters, max_iters, distance_func)
        self.clf = None

    def fit(self, X):
        n_samples, n_features = X.shape
        minsize = n_samples // self.n_clusters
        maxsize = (n_samples + self.n_clusters - 1) // self.n_clusters

        clf = KMeansConstrained(self.n_clusters, size_min=minsize,
                                size_max=maxsize)

        if minsize != maxsize:
            warnings.warn("Cluster minimum and maximum size are {} and {}, respectively".format(minsize, maxsize))

        clf.fit(X)

        self.clf = clf
        self.cluster_centers_ = self.clf.cluster_centers_
        self.labels_ = self.clf.labels_

    def predict(self, X):
        return self.clf.predict(X)

