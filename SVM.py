import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift

style.use('ggplot')

db_dir = 'db'

ticker = 'TSLA2016'
df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True, index_col = 0)

X = df["Close"].values
X = np.array([[k,X[k]] for k in range(len(X))])

np.random.seed(26)

# num_points = 1000
num_clusters = 5

# X, y = make_blobs(n_samples = num_points, n_features = 2, centers = num_clusters)

kmeans = KMeans(n_clusters = num_clusters, init='random', algorithm='full')
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

ax11 = plt.subplot2grid((3,2),(0,0), rowspan=1, colspan=1)
ax21 = plt.subplot2grid((3,2),(1,0), rowspan=1, colspan=1, sharex=ax11)
ax31 = plt.subplot2grid((3,2),(2,0), rowspan=1, colspan=1, sharex=ax11)
ax12 = plt.subplot2grid((3,2),(0,1), rowspan=1, colspan=1, sharex=ax11)
ax22 = plt.subplot2grid((3,2),(1,1), rowspan=1, colspan=1, sharex=ax11)
ax32 = plt.subplot2grid((3,2),(2,1), rowspan=1, colspan=1, sharex=ax11)

plt.figure(1)
ax11.set_title("Data")
ax11.scatter(X[:, 0], X[:, 1], c=labels)
ax11.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)


from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC

oneVSone = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X,labels)
a = oneVSone.predict(X)

# plt.figure(2)
ax21.set_title("One Vs One - Linear SVC")
ax21.scatter(X[:, 0], X[:, 1], c = a)

a = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X,labels).predict(X)
print(a)
ax31.set_title("One Vs Rest - Linear SVC")
ax31.scatter(X[:, 0], X[:, 1], c = a)

ax12.set_title("Data")
ax12.scatter(X[:, 0], X[:, 1], c=labels)
ax12.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)

a = OneVsOneClassifier(SVC(random_state=0)).fit(X,labels).predict(X)

# plt.figure(2)
ax22.set_title("One Vs One - SVC")
ax22.scatter(X[:, 0], X[:, 1], c = a)

a = OneVsRestClassifier(SVC(random_state=0)).fit(X,labels).predict(X)
ax32.set_title("One Vs Rest - SVC")
ax32.scatter(X[:, 0], X[:, 1], c = a)

# bandwidth = estimate_bandwidth(X)
# ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
# ms = ms.fit(X)
# labels_ms = ms.predict(X)


# x = np.random.rand(500,2)
#
# kmeans = KMeans(n_clusters = num_clusters, init='random', algorithm='full')
# kmeans = kmeans.fit(x)
# labels = kmeans.predict(x)
# C = kmeans.cluster_centers_
#
# plt.figure(2)
# ax11 = plt.subplot2grid((3,2),(0,0), rowspan=1, colspan=1)
# ax21 = plt.subplot2grid((3,2),(1,0), rowspan=1, colspan=1, sharex=ax11)
# ax31 = plt.subplot2grid((3,2),(2,0), rowspan=1, colspan=1, sharex=ax11)
# ax12 = plt.subplot2grid((3,2),(0,1), rowspan=1, colspan=1, sharex=ax11)
# ax22 = plt.subplot2grid((3,2),(1,1), rowspan=1, colspan=1, sharex=ax11)
# ax32 = plt.subplot2grid((3,2),(2,1), rowspan=1, colspan=1, sharex=ax11)
#
# # plt.figure(2)
# ax11.scatter(x[:, 0], x[:, 1], c=labels)
# ax11.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)
#
# oneVSone = OneVsOneClassifier(LinearSVC(random_state=0)).fit(x,labels)
# a = oneVSone.predict(x)
# b = np.array([a[k] if w else 5 for k,w in enumerate(a == labels)])
#
# ax21.scatter(x[:, 0], x[:, 1], c = a)
#
# # vet = np.random.rand(10,2)
# # a = oneVSone.predict(vet)
# # ax21.scatter(vet[:, 0], vet[:, 1], marker='*', c = a, s = 500)
#
# oneVSrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x,labels)
# a = oneVSrest.predict(x)
# b = np.array([a[k] if w else 5 for k,w in enumerate(a == labels)])
#
# ax31.scatter(x[:, 0], x[:, 1], c = a)
#
#
# ax12.scatter(x[:, 0], x[:, 1], c=labels)
# ax12.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=500)
#
# oneVSone = OneVsOneClassifier(SVC(random_state=0)).fit(x,labels)
# a = oneVSone.predict(x)
# b = np.array([a[k] if w else 5 for k,w in enumerate(a == labels)])
#
# ax22.scatter(x[:, 0], x[:, 1], c = a)
#
# oneVSrest = OneVsRestClassifier(SVC(random_state=0)).fit(x,labels)
# a = oneVSrest.predict(x)
# b = np.array([a[k] if w else 5 for k,w in enumerate(a == labels)])
#
# ax32.scatter(x[:, 0], x[:, 1], c = a)

# clf -> classifier
# clf = svm.SVC()
# clf.fit(X,labels)
#
# labels_svm = clf.predict(X)

# plt.figure(3)
# plt.scatter(X[:, 0], X[:, 1], c=labels_ms)
# plt.figure(3)
# plt.scatter(X[:, 0], X[:, 1], color=cm.rainbow(np.linspace(0, 1, len(X))))

plt.show()
