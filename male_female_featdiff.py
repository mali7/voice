import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

file1 = pd.read_csv('phase1_df.csv')
file2  = pd.read_csv('phase2_df.csv')
ismale1 = file1['isPatientMale'].tolist()
ismale2 = file2['isPatientMale'].tolist()
ismale = ismale1+ismale2

features = ['D words','P words','D words %','P words %','D words sd','P words sd','D turns','P turns','D turns %','P turns %','D wpt','P wpt','D compound_avg','D neg_avg','D neu_avg','D pos_avg','D compound_sd','D neg_sd','D neu_sd','D pos_sd','P compound_avg','P neg_avg','P neu_avg','P pos_avg','P compound_sd','P neg_sd','P neu_sd','P pos_sd']
for f in features:

	feat1 = file1[f].tolist()
	feat2 = file2[f].tolist()
	feat = feat1+feat2
	
	male = []
	female = []
	for i in range(len(ismale)):
		if ismale[i]==1: 
			male.append(feat[i])
		else:
			female.append(feat[i])
	p = scipy.stats.ttest_ind(male,female)[1]
	if p< 0.05:
		print("feat = ",f,' p = ',p, "male = ",np.mean(male)," female = ",np.mean(female))
		name = "male "+f+".png"
		bin_size = 10
		n, bins, patches = plt.hist(male, bin_size, normed=1, facecolor='green', alpha=0.5)
		plt.title('male mean = %f, P =  %f' % (np.mean(male),  p))
		plt.xlabel(f)
		# plt.show()
		plt.savefig(name)
		plt.clf()

		name = "female "+f+".png"
		n, bins, patches = plt.hist(female, bin_size, normed=1, facecolor='blue', alpha=0.5)
		plt.title('female mean = %f, P =  %f' % (np.mean(female),  p))
		plt.xlabel(f)
		# plt.show()
		plt.savefig(name)
		plt.clf()

		
		name = "male female "+f+".png"
		n, bins, patches = plt.hist(male, bin_size, normed=1, facecolor='green', alpha=0.5)
		n, bins, patches = plt.hist(female, bin_size, normed=1, facecolor='blue', alpha=0.5)
		plt.title('male mean = %f, female mean = %f, P =  %f' % (np.mean(male),np.mean(female),  p))
		plt.xlabel(f)
		# plt.show()
		plt.savefig(name)
		plt.clf()
		# exit(1)