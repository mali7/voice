import numpy as np
import pandas as pd
import csv
import scipy
from scipy import stats
def get_feat(type):
	file  = pd.read_csv('phase1_df.csv')
	p1_files = file['Filename']
	# isPatientMale	isDoctorMale

	feat = file[type].tolist()
	p1_dict = dict()
	for i in range(len(p1_files)):
		p1_dict[p1_files[i]] = feat[i]

			
	file  = pd.read_csv('phase2_df.csv')
	p2_files = file['Filename']
	feat = file[type].tolist()
	
	p2_dict = dict()
	for i in range(len(p2_files)):
		p2_dict[p2_files[i]] = feat[i]
	
	p2_dict.update(p1_dict)
	return p2_dict