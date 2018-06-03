import sys

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

class KSVMeans(KMeans):

    def __init__(self, n_clusters = 8, init = 'k-means++', n_init = 10, max_iter = 300, \
                 tol = 0.0001, precompute_distances = 'auto', verbose = 0, \
                 random_state = None, copy_x = True, n_jobs = 1, algorithm = 'auto'):

        # KMeans.__init__(self, n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, \
        #                 tol = tol, precompute_distances = precompute_distances, verbose = verbose, \
        #                 random_state = random_state, copy_x = copy_x, n_jobs = n_jobs, algorithm = algorithm)
        self.kmeans = KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, \
                             tol = tol, precompute_distances = precompute_distances, verbose = verbose, \
                             random_state = random_state, copy_x = copy_x, n_jobs = n_jobs, algorithm = algorithm)
        self.labels = None
        self.classifier = None

    def fit(self, X, y = None, _classifier = 'oneVSone'):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        classifier : string
            'oneVSone' or 'oneVsRest'
        """
        # kmeans = KMeans.fit(X,y)
        self.kmeans = self.kmeans.fit(X,y)

        if _classifier == 'oneVSone':
            self.classifier = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, self.kmeans.labels_)
        elif _classifier == 'oneVSrest':
            self.classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, self.kmeans.labels_)
        else:
            print("Invalid _classifier chosen")
            sys.exit()

        self.labels = self.classifier.predict(X)

        return self.classifier

    def predict(self, X):
        try:
            return self.classifier.predict(X)
        except AttributeError as e:
            print("This KSVMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
            sys.exit()
