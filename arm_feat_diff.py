import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyexcel_ods import get_data
import operator
from get_ata import get_ata_score


file  = pd.read_csv('phase2_df.csv')
arm = file['Arm'].tolist()

features = ['D words','P words','D words %','P words %','D words sd','P words sd','D turns','P turns','D turns %','P turns %','D wpt','P wpt','D compound_avg','D neg_avg','D neu_avg','D pos_avg','D compound_sd','D neg_sd','D neu_sd','D pos_sd','P compound_avg','P neg_avg','P neu_avg','P pos_avg','P compound_sd','P neg_sd','P neu_sd','P pos_sd']

for feat in features:
	f = file[feat].tolist()
	arm_one_ata = []
	arm_two_ata = []

	for i in range(len(arm)):
		if arm[i] == 1:
			arm_one_ata.append(f[i])
		else:
			arm_two_ata.append(f[i])
	
	p = scipy.stats.ttest_ind(arm_one_ata,arm_two_ata)[1]
	if p<0.05:
		print (feat , np.mean(arm_one_ata),np.mean(arm_two_ata) , p)
		
		name = "Arm = 1 "+feat+".png"
		bin_size = 20
		n, bins, patches = plt.hist(arm_one_ata, bin_size, normed=1, facecolor='green', alpha=0.5)
		plt.title('Arm = 1 mean = %f, P =  %f' % (np.mean(arm_one_ata),  p))
		plt.xlabel(feat)
		# plt.show()
		plt.savefig(name)
		plt.clf()

		name = "Arm = 2 "+feat+".png"
		n, bins, patches = plt.hist(arm_two_ata, bin_size, normed=1, facecolor='blue', alpha=0.5)
		plt.title('Arm = 2 mean = %f, P =  %f' % (np.mean(arm_two_ata),  p))
		plt.xlabel(feat)
		# plt.show()
		plt.savefig(name)
		plt.clf()

		
		name = "Arm 1, 2 "+feat+".png"
		n, bins, patches = plt.hist(arm_one_ata, bin_size, normed=1, facecolor='green', alpha=0.5)
		n, bins, patches = plt.hist(arm_two_ata, bin_size, normed=1, facecolor='blue', alpha=0.5)
		plt.title('Arm = 1 mean = %f, Arm = 2 mean = %f, P =  %f' % (np.mean(arm_one_ata),np.mean(arm_two_ata),  p))
		plt.xlabel(feat)
		# plt.show()
		plt.savefig(name)
		plt.clf()
		# exit(1)