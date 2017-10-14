import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabaz_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
def file_len(fname):
    with open(fname) as f:
        for t, l in enumerate(f):
            pass
    return t + 1
file  = open('stats_with_header.csv','r')
n_line = file_len("stats_with_header.csv")
x_axis = 40
y_axis = 28
line = file.readline().split(',')
x_axis_name = line[x_axis]
y_axis_name = line[y_axis]
print (x_axis_name,y_axis_name)
X = np.zeros([n_line-1,2])
Ratings = np.zeros([n_line-1,7])
for i in range(n_line-1):
	line = file.readline().split(',')
	X[i,0] = float(line[x_axis]) 
	X[i,1] = float(line[y_axis]) 
	Ratings[i,:] = [int(j) for j in line[47:]]
# print (X,X.shape)
X[:,0] = X[:,0]/np.max(X[:,0])
X[:,1] = X[:,1]/np.max(X[:,1])

kmeans = KMeans(init= 'k-means++', n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# print("number of estimated clusters : %d" % n_clusters_)

# plot results
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()
print(metrics.silhouette_score(X, labels, metric='euclidean'))
print(metrics.calinski_harabaz_score(X, labels))


colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    for j in range(7):
        for i in Ratings[my_members,j]:
            print (i,'\t',end="")
        print()
    for j in range(7):
        print(np.mean(Ratings[my_members,j]),'\t',end="")
    print(len(X[my_members,0]))
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.xlabel(x_axis_name)
plt.ylabel(y_axis_name)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()