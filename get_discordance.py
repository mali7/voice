# discDmindRaw	discPmindRaw	discDmind	discPmind	discDmind2	discPmind2	discDmind3	discPmind3
import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
def get_discordance():
	file  = pd.read_csv('phase1_df.csv')
	p1_files = file['Filename']
	discDmindRaw = np.nan_to_num(np.array(file['discDmindRaw']))
	discPmindRaw = np.nan_to_num(np.array(file['discPmindRaw']))
	discDmind2 = np.nan_to_num(np.array(file['discDmind2']))
	discPmind2 = np.nan_to_num(np.array(file['discPmind2']))
	discDmind3 = np.nan_to_num(np.array(file['discDmind3']))
	discPmind3 = np.nan_to_num(np.array(file['discPmind3']))
	

	up = np.array(np.abs([discDmindRaw,discPmindRaw,discDmind2,discPmind2,discDmind3,discPmind3])).T
	
	# up = np.array(np.abs([discDmindRaw,discPmindRaw])).T

	up = np.sum(np.nan_to_num(up),axis=1)
	p1_dict = dict()
	for i in range(len(p1_files)):
		p1_dict[p1_files[i]] = up[i]
	# print(len(p1_dict))
			
	file  = pd.read_csv('phase2_df.csv')
	p2_files = file['Filename']
	discDmindRaw = np.nan_to_num(np.array(file['discDmindRaw']))
	discPmindRaw = np.nan_to_num(np.array(file['discPmindRaw']))
	discDmind2 = np.nan_to_num(np.array(file['discDmind2']))
	discPmind2 = np.nan_to_num(np.array(file['discPmind2']))
	discDmind3 = np.nan_to_num(np.array(file['discDmind3']))
	discPmind3 = np.nan_to_num(np.array(file['discPmind3']))
	

	up = np.array(np.abs([discDmindRaw,discPmindRaw,discDmind2,discPmind2,discDmind3,discPmind3])).T
	
	
	up = np.sum(np.nan_to_num(up),axis=1)
	p2_dict = dict()
	for i in range(len(p2_files)):
		p2_dict[p2_files[i]] = up[i]
	files, up1 = zip(*p1_dict.items())

	p2_dict.update(p1_dict)
	return p2_dict