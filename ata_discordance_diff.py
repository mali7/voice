from get_ata import get_ata_score
from get_discordance import get_discordance
import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


discordance_dict = get_discordance()
lowest_p = 1
best_rate = 0 

for rate in range(10):
	# r = 0.1+ (rate/100)
	r = rate/10
	ata_dict = get_ata_score(r)

	keys, notUse = zip(*discordance_dict.items())
	notUse, values = zip(*ata_dict.items())

	# print(np.median(values))
	list_low_ata = []
	list_high_ata = []
	for key in keys:
		if (ata_dict[key]>np.median(values)):
			list_high_ata.append(discordance_dict[key])
		else:
			list_low_ata.append(discordance_dict[key])
	p = scipy.stats.ttest_ind(list_high_ata,list_low_ata)[1]
	if (p<lowest_p):
		lowest_p = p
		best_rate = r
	print("r = ",r," n high ata=",len(list_high_ata)," mean = ", np.mean(list_high_ata),"n low ata=",len(list_low_ata)," mean = ",np.mean(list_low_ata))
	print(scipy.stats.ttest_ind(list_high_ata,list_low_ata))
	new_list = list_low_ata+ list_high_ata
	# print (np.mean(list_high_ata)-np.mean(list_low_ata) / np.std(new_list))
	

	
	
ata_dict = get_ata_score(best_rate)
keys, notUse = zip(*discordance_dict.items())
notUse, values = zip(*ata_dict.items())

# print(np.median(values))
list_low_ata = []
list_high_ata = []
for key in keys:
	if (ata_dict[key]>np.mean(values)):
		list_high_ata.append(discordance_dict[key])
	else:
		list_low_ata.append(discordance_dict[key])

print("r = ",best_rate," n high ata=",len(list_high_ata)," mean = ", np.mean(list_high_ata),"n low ata=",len(list_low_ata)," mean = ",np.mean(list_low_ata))
print(scipy.stats.ttest_ind(list_high_ata,list_low_ata))
new_list = list_low_ata+ list_high_ata


n_tot = len(list_high_ata) + len(list_low_ata)
x_var = np.var(list_high_ata)
y_var = np.var(list_low_ata)
pooled_var = (len(list_high_ata)*x_var + len(list_low_ata)*y_var) / n_tot

if(pooled_var ==0):
	cohens_d = np.nan
else:
	cohens_d = (np.mean(list_high_ata)-np.mean(list_low_ata)) / np.sqrt(pooled_var)

# print (np.mean(list_high_ata)-np.mean(list_low_ata) / np.std(new_list))
print(cohens_d)

bin_size = 10
p = scipy.stats.ttest_ind(list_high_ata,list_low_ata)[1]
n, bins, patches = plt.hist(list_high_ata, bin_size, normed=1, facecolor='green', alpha=0.5)
n, bins, patches = plt.hist(list_low_ata, bin_size, normed=1, facecolor='blue', alpha=0.5)
plt.title('high mean = %f, low mean = %f, P =  %f' % (np.mean(list_high_ata),np.mean(list_low_ata),  p))

# plt.title('high mean = %f, low mean = %f, P =  %f , cohens_d = %f' % (np.mean(list_high_ata),np.mean(list_low_ata),  p, cohens_d))
plt.xlabel('ATA + Med Name Scores')
plt.ylabel('Percentage of interactions')
# plt.show()
plt.savefig('high-low-ata.png')
plt.clf()