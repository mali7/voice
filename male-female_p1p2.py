import numpy as np
import pandas as pd
import csv
import scipy
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
	
file1 = pd.read_csv('phase1_df.csv')
file2  = pd.read_csv('phase2_df.csv')
ismale1 = file1['isPatientMale'].tolist()
ismale2 = file2['isPatientMale'].tolist()
ismale = ismale1+ismale2
print ("ismale", len(ismale),np.sum(ismale))

# exit(1)
# file  = pd.read_csv('phase1_df.csv')
# p_up1 = np.nan_to_num(np.array(file['p0_up1']))
# p_up2 = np.nan_to_num(np.array(file['p0_up2']))
# p_up3 = np.nan_to_num(np.array(file['p0_up3']))
# d_up1 = np.nan_to_num(np.array(file['d1_up1']))
# d_up2 = np.nan_to_num(np.array(file['d1_up2']))
# d_up3 = np.nan_to_num(np.array(file['d1_up3']))
# up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T
# up = np.sum(np.nan_to_num(up),axis=1)
# # print (up.shape)
# file  = pd.read_csv('phase2_df.csv')
# p_up1 = np.nan_to_num(np.array(file['p0_up1']))
# p_up2 = np.nan_to_num(np.array(file['p0_up2']))
# p_up3 = np.nan_to_num(np.array(file['p0_up3']))
# d_up1 = np.nan_to_num(np.array(file['d1_up1']))
# d_up2 = np.nan_to_num(np.array(file['d1_up2']))
# d_up3 = np.nan_to_num(np.array(file['d1_up3']))
# up2 = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T
# up2 = np.sum(np.nan_to_num(up2),axis=1)/2
# # print (up2.shape)
# up_all = np.append(up,up2)
# # print (up.shape)

######################new discordance#####################################

file  = pd.read_csv('phase1_df.csv')
# p1_files = file['Filename']
discDmindRaw = np.nan_to_num(np.array(file['discDmindRaw']))
discPmindRaw = np.nan_to_num(np.array(file['discPmindRaw']))
discDmind2 = np.nan_to_num(np.array(file['discDmind2']))
discPmind2 = np.nan_to_num(np.array(file['discPmind2']))
discDmind3 = np.nan_to_num(np.array(file['discDmind3']))
discPmind3 = np.nan_to_num(np.array(file['discPmind3']))

up = np.array(np.abs([discDmindRaw,discPmindRaw,discDmind2,discPmind2,discDmind3,discPmind3])).T
up = np.sum(np.nan_to_num(up),axis=1)		
file  = pd.read_csv('phase2_df.csv')
p2_files = file['Filename']
discDmindRaw = np.nan_to_num(np.array(file['discDmindRaw']))
discPmindRaw = np.nan_to_num(np.array(file['discPmindRaw']))
discDmind2 = np.nan_to_num(np.array(file['discDmind2']))
discPmind2 = np.nan_to_num(np.array(file['discPmind2']))
discDmind3 = np.nan_to_num(np.array(file['discDmind3']))
discPmind3 = np.nan_to_num(np.array(file['discPmind3']))


up2 = np.array(np.abs([discDmindRaw,discPmindRaw,discDmind2,discPmind2,discDmind3,discPmind3])).T
up2 = np.sum(np.nan_to_num(up2),axis=1)
up_all = np.append(up,up2)





features = ['D words','P words','D words %','P words %','D words sd','P words sd','D turns','P turns','D turns %','P turns %','D wpt','P wpt','D compound_avg','D neg_avg','D neu_avg','D pos_avg','D compound_sd','D neg_sd','D neu_sd','D pos_sd','P compound_avg','P neg_avg','P neu_avg','P pos_avg','P compound_sd','P neg_sd','P neu_sd','P pos_sd']

for i in features:
	for j in features[features.index(i)+1:]:
		up = up_all
		file  = pd.read_csv('phase1_df.csv')
		x_axis = np.array(file[i])
		y_axis = np.array(file[j])
		file  = pd.read_csv('phase2_df.csv')
		x_axis= np.append(x_axis,np.array(file[i]))
		y_axis= np.append(y_axis,np.array(file[j]))
		x_axis_name = i
		y_axis_name = j
		# print (x_axis_name,y_axis_name)
		X = [x_axis,y_axis]
		X = np.array(X).T
		# print (X.shape)
		X[:,0] = X[:,0]/np.max(X[:,0])
		X[:,1] = X[:,1]/np.max(X[:,1])
		
		#############male female division######################
		males = []
		for l in range(len(ismale)):
			# print (ismale[l])
			if ismale[l] == 1:
				males.append(l)
		males = np.array(males)
		X = np.delete(X,males,0)
		up = np.delete(up,males,0)
		
		
		kmeans = KMeans(init= 'k-means++', n_clusters=3, random_state=0).fit(X)
		labels = kmeans.labels_
		cluster_centers = kmeans.cluster_centers_
		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)
		three_cluster_silhouette = metrics.silhouette_score(X, labels, metric='euclidean')

		kmeans = KMeans(init= 'k-means++', n_clusters=2, random_state=0).fit(X)
		labels = kmeans.labels_
		cluster_centers = kmeans.cluster_centers_
		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)
		two_cluster_silhouette = metrics.silhouette_score(X, labels, metric='euclidean')

		if (two_cluster_silhouette>=three_cluster_silhouette):

			import matplotlib.pyplot as plt
			from itertools import cycle

			plt.figure(1)
			plt.clf()

			colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
			m = 0
			t = []
			for k, col in zip(range(n_clusters_), colors):
				my_members = labels == k
				cluster_center = cluster_centers[k]
				# t.append(Ratings[my_members])
				t.append(up[my_members])
				m+=1
				plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
				plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
						 markeredgecolor='k', markersize=14,label = np.around(np.mean(t[m-1]),decimals=2))
			p = scipy.stats.ttest_ind(t[0],t[1])[1]
			if p < 0.05:
				name ="Male "+ x_axis_name+" vs "+y_axis_name
				print(name, np.mean(t[0]),np.mean(t[1]),p)
				plt.xlabel(x_axis_name)
				plt.ylabel(y_axis_name)
				plt.legend(loc='upper right')
				plt.title('P =  %f' % p)
				plt.savefig(name)