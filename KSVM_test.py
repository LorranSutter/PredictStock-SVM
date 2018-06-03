import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from KSVMeans import KSVMeans

style.use('ggplot')

# ---------- BEGIN Parameters ---------- #

db_dir = 'db'

ticker = 'TSLA2016'
df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True, index_col = 0)

X = df["Close"].values
X = np.array([[k,X[k]] for k in range(len(X))])

num_clusters = 3

ax11 = plt.subplot2grid((3,2),(0,0), rowspan=1, colspan=1)
ax21 = plt.subplot2grid((3,2),(1,0), rowspan=1, colspan=1, sharex=ax11)
ax31 = plt.subplot2grid((3,2),(2,0), rowspan=1, colspan=1, sharex=ax11)
ax12 = plt.subplot2grid((3,2),(0,1), rowspan=1, colspan=1, sharex=ax11)
ax22 = plt.subplot2grid((3,2),(1,1), rowspan=1, colspan=1, sharex=ax11)
ax32 = plt.subplot2grid((3,2),(2,1), rowspan=1, colspan=1, sharex=ax11)

# ---------- END Parameters ---------- #



# ---------- BEGIN KMeans + OneVsOne or OneVSRest ---------- #

kmeans = KMeans(n_clusters = num_clusters, init='random', algorithm='full')
kmeans = kmeans.fit(X)
labels_kmeans = kmeans.predict(X)
C = kmeans.cluster_centers_

oneVSone = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, labels_kmeans)
oneVrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, labels_kmeans)
labels_OvO = oneVSone.predict(X)
labels_OvR = oneVrest.predict(X)

# ---------- END KMeans + OneVsOne or OneVSRest ---------- #



# ---------- BEGIN KSVMeans ---------- #

ksvmeans = KSVMeans(n_clusters = num_clusters, init='random', algorithm='full')

# ksvmeans_OvO = ksvmeans.fit(X, _classifier = 'oneVSone')
labels_KSVM_OvO = ksvmeans.predict(X)

ksvmeans_OvR = ksvmeans.fit(X, _classifier = 'oneVSrest')
labels_KSVM_OvR = ksvmeans.predict(X)

# ---------- END KSVMeans ---------- #



# ---------- BEGIN KMeans + OneVsOne or OneVSRest PLOT ---------- #

# plt.figure(1)
ax11.set_title("Data")
ax11.scatter(X[:, 0], X[:, 1], c = labels_kmeans)
ax11.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)

ax21.set_title("One Vs One - Linear SVC")
ax21.scatter(X[:, 0], X[:, 1], c = labels_OvO)

ax31.set_title("One Vs Rest - Linear SVC")
ax31.scatter(X[:, 0], X[:, 1], c = labels_OvR)

# ---------- END KMeans + OneVsOne or OneVSRest PLOT ---------- #



# ---------- BEGIN KSVMeans PLOT ---------- #

ax12.set_title("Data")
ax12.scatter(X[:, 0], X[:, 1], c = labels_kmeans)
ax12.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)

ax22.set_title("One Vs One - Linear SVC")
ax22.scatter(X[:, 0], X[:, 1], c = labels_KSVM_OvO)

ax32.set_title("One Vs Rest - Linear SVC")
ax32.scatter(X[:, 0], X[:, 1], c = labels_KSVM_OvR)

# ---------- END KSVMeans PLOT ---------- #

plt.show()
