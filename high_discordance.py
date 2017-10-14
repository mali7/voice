import numpy as np
import pandas as pd
file  = pd.read_csv('phase2_df.csv')
p_up1 = np.nan_to_num(np.array(file['p0_up1']))
p_up2 = np.nan_to_num(np.array(file['p0_up2']))
p_up3 = np.nan_to_num(np.array(file['p0_up3']))
d_up1 = np.nan_to_num(np.array(file['d1_up1']))
d_up2 = np.nan_to_num(np.array(file['d1_up2']))
d_up3 = np.nan_to_num(np.array(file['d1_up3']))
# up = np.array([np.abs(p_up1-d_up1),np.abs(p_up2-d_up2),np.abs(p_up3-d_up3),np.abs(p_up2-d_up1),np.abs(p_up1-d_up2)]).T
up = np.array([np.abs(p_up2-d_up1)]).T
up = np.sum(np.nan_to_num(up),axis=1)
fileNames = file['Filename']
for i in range(len(fileNames)):
	print (fileNames[i],up[i])
d_wpt = file['D wpt']
print (np.array(d_wpt), np.std(np.array(d_wpt)))