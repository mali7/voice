import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyexcel_ods import get_data
import operator

def get_lecturing(rate = 1):
	window = 10
	file  = pd.read_csv('all_w_sent.csv',encoding = "ISO-8859-1")
	file_names = file['Filename'].tolist()
	speaker = file['Speaker'].tolist()
	text = file['Text'].tolist()
	new_dict = dict()
	files = np.unique(file_names)
	for i in files:
		new_dict[i] = 0
	d_mean = [0]
	for i in range(len(text)-window):
		mean = np.mean([len(j) for j in text[i:i+window]])
		sd = np.std([len(j) for j in text[i:i+window]])
		# print (mean,sd)
		d_mean = [0]
		for j in range(window):
			if speaker[i+j] =='D':
				d_mean.append(len(text[i+j]))
		d_mean = np.mean(d_mean)
		diff = (d_mean-mean)/sd
		if diff > rate:
			new_dict[file_names[i]] +=1
			
	return new_dict