import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyexcel_ods import get_data
import operator

def get_ata_score(rate = 1):
	
	data = get_data("medicineList.ods")
	med_names = dict()
	for i in data:
		for j in data[i]:
			for k in j:
				med_names[k.lower()] = 0
	del med_names['at']
	del med_names['h']
	del med_names['th']
	
	file  = pd.read_csv('all_w_sent.csv',encoding = "ISO-8859-1")
	file_names = file['Filename'].tolist()
	speaker = file['Speaker'].tolist()
	text = file['Text'].tolist()
	new_dict = dict()
	files = np.unique(file_names)
	for i in files:
		new_dict[i] = 0
		
	for i in range(len(text)):
		for word in text[i].split():
			if word.lower() in med_names.keys():
				med_names[word.lower()] +=1
				new_dict[file_names[i]] +=1
				
				
	Qlist = ['what','when','why','which','who','how','whose','whom']
	
	for i in range(len(text)):
		for w in Qlist:
			if w.lower() in text[i]:
				new_dict[file_names[i]] -=1
	
	for i in range(len(text)):
		if 'understand' in text[i] and speaker[i] == 'D' and '?' in text[i]:
			new_dict[file_names[i]] +=1
	
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
					
					
				set1 = set()
				set2 = set()
				for words in d_list[n-1].split(): 
					set1.add(words)
				# print(set1)
				for words in d_list[n-2].split(): 
					set2.add(words) 
				set1.intersection_update(set2)

				new_dict[file_names[i]] += len(set1)*rate
				# exit(1)
		cur_file = file_names[i]
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
			new_dict[file_names[i]] += len(d_turn_set)*rate
			# print (d_turn_set)
			d_turn_set = set()
			p_turn_set = set()
	return new_dict