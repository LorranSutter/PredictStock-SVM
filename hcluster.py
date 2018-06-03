import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as hcluster

num_points = 1000
num_clusters = 5

X, y = make_blobs(n_samples = num_points, n_features = 2, centers = num_clusters)

# generate 3 clusters of each around 100 points and one orphan point
# N=100
# data = numpy.random.randn(3*N,2)
# data[:N] += 5
# data[-N:] += 10
# data[-1:] -= 20

# clustering
thresh = 1.5
clusters = hcluster.fclusterdata(X, thresh, criterion="distance")

# plotting
# plt.scatter(*numpy.transpose(data), c=clusters)
# plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.show()
