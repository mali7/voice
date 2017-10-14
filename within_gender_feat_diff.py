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
from get_is_male import get_is_male
from get_discordance import get_discordance
from get_feat import get_feat

is_male_dict = get_is_male('isPatientMale')
discordance_dict = get_discordance()
files, values = zip(*discordance_dict.items())
features = ['D words','P words','D words %','P words %','D words sd','P words sd','D turns','P turns','D turns %','P turns %','D wpt','P wpt','D compound_avg','D neg_avg','D neu_avg','D pos_avg','D compound_sd','D neg_sd','D neu_sd','D pos_sd','P compound_avg','P neg_avg','P neu_avg','P pos_avg','P compound_sd','P neg_sd','P neu_sd','P pos_sd']
for feat in features:
	feat_dict = get_feat(feat)
	high_dis_feats = []
	low_dis_feats = []
	for f in files:
		if is_male_dict[f] == 1:
			if discordance_dict[f] > np.median(values):
				high_dis_feats.append(feat_dict[f])
			else:
				low_dis_feats.append(feat_dict[f])
	p = scipy.stats.ttest_ind(high_dis_feats,low_dis_feats)[1]
	if p< 0.05:
		print ("feature = ",feat," p= ",p," high_mean = ",np.mean(high_dis_feats)," low mean = ",np.mean(low_dis_feats))