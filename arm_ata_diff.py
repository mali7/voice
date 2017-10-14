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

new_dict = get_ata_score()
keys, values = zip(*new_dict.items())

file  = pd.read_csv('phase2_df.csv')
arm = file['Arm'].tolist()
filename = file['Filename'].tolist()

arm_one_ata = []
arm_two_ata = []

for i in range(len(arm)):
	if arm[i] == 1:
		arm_one_ata.append(new_dict[filename[i]])
	else:
		arm_two_ata.append(new_dict[filename[i]])
		
# file  = pd.read_csv('phase1_df.csv')
# filename = file['Filename'].tolist()
# for i in filename:
	# arm_one_ata.append(new_dict[i])
	
print (np.mean(arm_one_ata),np.mean(arm_two_ata))
print(scipy.stats.ttest_ind(arm_one_ata,arm_two_ata))

# list_low_ata = []
# list_high_ata = []
# for i in files:
	# if (new_dict[i]>np.median(values)):
		# list_high_ata.append(p2_dict[i])
	# else:
		# list_low_ata.append(p2_dict[i])

# print("n high ata=",len(list_high_ata)," mean = ", np.mean(list_high_ata),"n low ata=",len(list_low_ata)," mean = ",np.mean(list_low_ata))
# print(scipy.stats.ttest_ind(list_high_ata,list_low_ata))

# ########################hist###################################
# n, bins, patches = plt.hist(up, 20, normed=1, facecolor='green', alpha=0.5)
# plt.title("Phase2 discordance")
# # plt.show()
# plt.savefig('Phase2 discordance.png')
# plt.clf()
# n, bins, patches = plt.hist(up1, 20, normed=1, facecolor='green', alpha=0.5)
# plt.title("Phase1 discordance")
# # plt.show()
# plt.savefig('Phase1 discordance.png')
# plt.clf()
# n, bins, patches = plt.hist(values, 20, normed=1, facecolor='green', alpha=0.5)
# plt.title("Ask Tell Ask Score Histogram")
# plt.xlabel("ATA Score")
# plt.ylabel("Percentage of interaction")
# # plt.show()
# plt.savefig('Ask Tell Ask Score.png')