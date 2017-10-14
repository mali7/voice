import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
def get_up():
	file  = pd.read_csv('phase1_df.csv')
	p1_files = file['Filename']
	p_up1 = np.nan_to_num(np.array(file['p0_up1']))
	p_up2 = np.nan_to_num(np.array(file['p0_up2']))
	p_up3 = np.nan_to_num(np.array(file['p0_up3']))
	d_up1 = np.nan_to_num(np.array(file['d1_up1']))
	d_up2 = np.nan_to_num(np.array(file['d1_up2']))
	d_up3 = np.nan_to_num(np.array(file['d1_up3']))
	# up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

	up = np.array([np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

	# up = np.array([np.abs(p_up2-d_up1)]).T

	up = np.sum(np.nan_to_num(up),axis=1)
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
	# up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

	up = np.array([np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T

	# up = np.array([np.abs(p_up2-d_up1)]).T

	up = np.sum(np.nan_to_num(up),axis=1)/2
	p2_dict = dict()
	for i in range(len(p2_files)):
		p2_dict[p2_files[i]] = up[i]
	# print(len(p2_dict))
	files, up1 = zip(*p1_dict.items())

	p2_dict.update(p1_dict)
	return p2_dict