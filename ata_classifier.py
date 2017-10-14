import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors, datasets
from sklearn import linear_model
from pyexcel_ods import get_data
import operator
data = get_data("medicineList.ods")
med_names = dict()
for i in data:
	for j in data[i]:
		for k in j:
			med_names[k.lower()] = 0
del med_names['at']

file  = pd.read_csv('all_w_sent.csv',encoding = "ISO-8859-1")
file_names = file['Filename'].tolist()
speaker = file['Speaker'].tolist()
text = file['Text'].tolist()
list_of_dicts = []
new_dict = dict()
files = np.unique(file_names)
for i in files:
	new_dict[i] = 0
	
for i in range(len(text)):
	for word in text[i].split():
		if word.lower() in med_names.keys():
			med_names[word.lower()] +=1
			new_dict[file_names[i]] +=1

list_of_dicts.append(new_dict)	
new_dict = dict()
files = np.unique(file_names)
for i in files:
	new_dict[i] = 0			

for i in range(len(text)):
	if 'understand' in text[i] and speaker[i] == 'D' and '?' in text[i]:
		new_dict[file_names[i]] +=1


list_of_dicts.append(new_dict)	
new_dict = dict()
files = np.unique(file_names)
for i in files:
	new_dict[i] = 0
	
d_list = []
cur_file = ""
for i in range(len(text)):
	if speaker[i] == 'D':
		if len(d_list)>0 and file_names[i] != cur_file:
			d_list = []
		d_list.append(text[i])
		# print (d_list,text[i])
		if len(d_list)>=3:
			n = len(d_list)
			if '?' in d_list[n-1] and '?' in d_list[n-3] and '?' not in d_list[n-2]:
				new_dict[file_names[i]] +=1
	cur_file = file_names[i]
list_of_dicts.append(new_dict)	
new_dict = dict()
files = np.unique(file_names)
for i in files:
	new_dict[i] = 0			
				
d_list = []
cur_file = ""
for i in range(len(text)):
	if speaker[i] == 'D':
		if len(d_list)>0 and file_names[i] != cur_file:
			d_list = []
		d_list.append(text[i])
		# print (d_list,text[i])
		if len(d_list)>=3:
			n = len(d_list)				
			set1 = set()
			set2 = set()
			for words in d_list[n-1].split(): 
				set1.add(words)
			# print(set1)
			for words in d_list[n-2].split(): 
				set2.add(words) 
			set1.intersection_update(set2)
			# print(set1)
			new_dict[file_names[i]] +=  len(set1)
			# exit(1)
	cur_file = file_names[i]
	
list_of_dicts.append(new_dict)	
new_dict = dict()
files = np.unique(file_names)
for i in files:
	new_dict[i] = 0
	
d_turn_set = set()
p_turn_set = set()
for i in range(len(text)):
	if speaker[i] == 'D':
		d_turn_set = set()
		for w in text[i].split():
			d_turn_set.add(w ) 
		# print (d_turn_set)
	elif speaker[i] == 'P':
		p_turn_set = set()
		for w in text[i].split():
			p_turn_set.add(w ) 
		# print (p_turn_set)
	if len(d_turn_set)>0 and len(p_turn_set)>0:
		d_turn_set.intersection_update(p_turn_set)
		new_dict[file_names[i]] += len(d_turn_set)
		# print (d_turn_set)
		d_turn_set = set()
		p_turn_set = set()

list_of_dicts.append(new_dict)

keys, values = zip(*new_dict.items())
# print (keys,values)	
print(len(list_of_dicts))
# for i in keys:
	# print (i,'\t',new_dict[i])


#####################reading up discordance################################
file  = pd.read_csv('phase1_df.csv')
p1_files = file['Filename']
p_up1 = np.nan_to_num(np.array(file['p0_up1']))
p_up2 = np.nan_to_num(np.array(file['p0_up2']))
p_up3 = np.nan_to_num(np.array(file['p0_up3']))
d_up1 = np.nan_to_num(np.array(file['d1_up1']))
d_up2 = np.nan_to_num(np.array(file['d1_up2']))
d_up3 = np.nan_to_num(np.array(file['d1_up3']))
up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

# up = np.array([np.abs(p_up1-d_up1)]).T
# up = np.sum(np.nan_to_num(up),axis=1)
p1_dict = dict()
for i in range(len(p1_files)):
	p1_dict[p1_files[i]] = up[i]
# print(len(p1_dict))
		
file  = pd.read_csv('phase2_df.csv')
p2_files = file['Filename']
p_up1 = np.nan_to_num(np.array(file['p0_up1']))
p_up2 = np.nan_to_num(np.array(file['p0_up2']))
p_up3 = np.nan_to_num(np.array(file['p0_up3']))
d_up1 = np.nan_to_num(np.array(file['d1_up1']))
d_up2 = np.nan_to_num(np.array(file['d1_up2']))
d_up3 = np.nan_to_num(np.array(file['d1_up3']))
up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

# up = np.array([np.abs(p_up1-d_up1)]).T

# up = np.sum(np.nan_to_num(up),axis=1)
p2_dict = dict()
for i in range(len(p2_files)):
	p2_dict[p2_files[i]] = up[i]/2
# print(len(p2_dict))
p2_dict.update(p1_dict)

files, up = zip(*p2_dict.items())
# list_low_ata = []
# list_high_ata = []
# for i in files:
	# if (new_dict[i]>np.mean(values)):
		# list_high_ata.append(p2_dict[i])
	# else:
		# list_low_ata.append(p2_dict[i])

# print("n high ata=",len(list_high_ata)," mean = ", np.mean(list_high_ata),"n low ata=",len(list_low_ata)," mean = ",np.mean(list_low_ata))
# print(scipy.stats.ttest_ind(list_high_ata,list_low_ata))

########################classifier###################################
X = np.zeros([len(files),len(list_of_dicts)])
Y = []
j = 0
for i in files:
	for feat in range(len(list_of_dicts)):
		X[j,feat] = list_of_dicts[feat][i]
		# X[j,1] = list_of_dicts[1][i]
		# X[j,2] = list_of_dicts[2][i]
		# X[j,3] = list_of_dicts[3][i]
	Y.append( p2_dict[i])
	j+=1
# print(X,Y)
Y = np.array(Y)

for i in range(Y.shape[1]):
	med = np.median(Y[:,i])
	Y[:,i] = [1 if j > med else 0 for j in Y[:,i] ]
# print (Y)
X = preprocessing.scale(X)



for i in range (Y.shape[1]):
	
	clf = svm.SVC(kernel='rbf',max_iter = 6000,C=2,gamma=1)
	# clf = linear_model.Lasso(alpha=0.1)
	# clf = neighbors.KNeighborsClassifier(3, weights='distance') 
	ans = cross_val_predict(clf, X, Y[:,i], cv=10)
	f1 = f1_score(Y[:,i], ans)
	print("F1 score = ",f1)
	print("Accuracy = ",accuracy_score(Y[:,i], ans))
	print(classification_report(Y[:,i], ans))