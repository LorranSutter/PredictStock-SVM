import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from mpl_toolkits.mplot3d import Axes3D

import Indicators as ind

db_dir = 'db'

def remap(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def cluster_id_min_dist(dists, point, num_clusters):
    closer = np.inf
    closer_id = 0
    for k in range(num_clusters):
        if dists[k,point] < closer:
            closer = dists[k,point]
            closer_id = k
    return closer_id

plot = False
num_points = 1000
num_clusters = 3
changed = True

X, y = make_blobs(n_samples = num_points, n_features = 2, centers = num_clusters)

ticker = 'TSLA'
df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True, index_col = 1)
df['RSI_10'] = ind.RSI(df,10)
# old_X = np.array([[k.timestamp(),c] for k,c in zip(df.index, df['Close'].values)])
df = df.dropna()
old_X = np.array([[k.timestamp(),c,rsi] for k,c,rsi in zip(df.index, df['Close'].values, df['RSI_10'].values)])

# old_X = old_X[np.random.choice(old_X.shape[0], len(old_X)//10, replace=False), :]

# X = old_X.copy()
# X[:,0] = np.random.rand(old_X.shape[0])*(385-92)+92

# X = np.random.rand(old_X.shape[0], 3)*(385-92)+92
# X[:,:-1] = old_X
# X[:,-1] = df['Volume'].values

labels = [-1 for k in range(num_points)]
# V = np.array([X[k] for k in np.random.choice(range(num_points), size = num_clusters, replace = False)])
# print(V)
C = np.array([0 for k in range(num_clusters)])

# while changed:
#     changed = False
#
#     dists = np.matrix([[0.0 for k in range(num_points)] for w in range(num_clusters)])
#     for k in range(num_clusters):
#         for w in range(num_points):
#             dists[k,w] = distance(V[k],X[w])
#
#     C = np.array([0 for k in range(num_clusters)])
#     for k in range(num_points):
#         new_c = cluster_id_min_dist(dists, k, num_clusters)
#         C[new_c] += 1
#         if new_c != labels[k]:
#             changed = True
#         labels[k] = new_c
#
#     # print(C)
#
#     for k in range(num_clusters):
#         V[k] = 1/C[k] * sum([X[w] for w in range(num_points) if labels[w] == k])
#
# ax1 = plt.subplot2grid((2,1),(0,0), rowspan=1, colspan=1)
# ax2 = plt.subplot2grid((2,1),(1,0), rowspan=1, colspan=1)

if plot:
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')


# plt.figure(1)
# plt.scatter(X[:,0],X[:,1], c = labels)
# plt.scatter(V[:,0],V[:,1], marker='*', c='#050505', s=500)

# plt.show()


# --- Comparison - native K-Means --- #

if plot:
    kmeans = KMeans(n_clusters = num_clusters, init='random', algorithm='full')
    kmeans = kmeans.fit(old_X)
    labels = kmeans.predict(old_X)
    C1 = kmeans.cluster_centers_

    # plt.figure(2)
    ax1.set_title("2d data")
    ax1.scatter(old_X[:, 0], old_X[:, 2], old_X[:, 1], c=labels)
    ax1.scatter(C1[:, 0], C1[:, 2], C1[:, 1], marker='*', c=range(len(C1)), s=500)

if plot:
    kmeans = KMeans(n_clusters = num_clusters, init='random', algorithm='full')
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    C2 = kmeans.cluster_centers_

    # plt.figure(2)
    ax2.set_title("3d data")
    # ax2.scatter(old_X[:, 0], X[:, 1], c = labels)
    ax2.scatter(X[:, 0], X[:, 2], X[:, 1], c = labels)
    # ax2.scatter(remap(C2[:, 0], 92, 385, min(C1[:,0]), max(C1[:,0])), C2[:, 1], marker='*', c=range(len(C2)), s=500)
    ax2.scatter(C2[:, 0], C2[:, 2], C2[:, 1], marker='*', c=range(len(C2)), s=500)

if plot:
    plt.show()


# --- Comparison - tensorflow K-Means --- #

# import tensorflow as tf

# def input_fn():
#   return tf.train.limit_epochs(tf.convert_to_tensor(X, dtype=tf.float32), num_epochs=1)
#
# tf_kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)

# train
# num_iterations = 10
# previous_centers = None
# for _ in range(num_iterations):
#   tf_kmeans.train(input_fn)
#   cluster_centers = tf_kmeans.cluster_centers()
#   if previous_centers is not None:
#     print('delta:', cluster_centers - previous_centers)
#   previous_centers = cluster_centers
#   print('score:', tf_kmeans.score(input_fn))
# print('cluster centers:', cluster_centers)

# map the input points to their clusters
# cluster_indices = list(tf_kmeans.predict_cluster_index(input_fn))
# for i, point in enumerate(X):
#   cluster_index = cluster_indices[i]
#   center = cluster_centers[cluster_index]
#   print('point:', point, 'is in cluster', cluster_index, 'centered at', center)

# plt.figure(3)
# ax2.scatter(X[:, 0], X[:, 1], c=cluster_indices)
# ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', c='#050505', s=500)

# plt.show()
