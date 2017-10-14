import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from get_is_male import get_is_male
from get_up import get_up
from get_discordance import get_discordance
is_male_dict = get_is_male('isPatientMale')
up_dict = get_up()
discordance_dict = get_discordance()
print (len(up_dict),len(is_male_dict), discordance_dict)

keys, values = zip(*discordance_dict.items())

male = []
female = []
for key in keys:
	if is_male_dict[key] == 1:
		male.append(discordance_dict[key])
	else:
		female.append(discordance_dict[key])

print (np.mean(male),np.mean(female),len(male),len(female),np.max(male),np.max(female))


bin_size = 10
p = scipy.stats.ttest_ind(male,female)
print(p)
n, bins, patches = plt.hist(male, bin_size, normed=1, facecolor='blue', alpha=0.5)
plt.title("male discordance")
plt.xlabel('Discordance')
plt.ylabel('Percentage of Doctors')
plt.savefig('male discordance.png')
plt.clf()

n, bins, patches = plt.hist(female, bin_size, normed=1, facecolor='green', alpha=0.5)
plt.title("female discordance")
plt.xlabel('Discordance')
plt.ylabel('Percentage of Doctors')
plt.savefig('female discordance.png')
plt.clf()

# n, bins, patches = plt.hist(female, 5, normed=1, facecolor='blue', alpha=0.5)
# plt.title("female discordance")
# # plt.show()
# plt.savefig('female discordance.png')
# plt.clf()