# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:17:16 2019

@author: Dacong
"""

import os
from glob import glob
#from sklearn.model_selection import cross_val_score
#from sklearn.datasets import make_blobs

import numpy as np

subjs=['207','208','209','210','211']

#process chewing features and labels
chewing_data=[]
for subj in subjs:
    PATH=os.path.join('camera chewing', subj+' chewing', 'MFCC', '*.csv')
    data = np.vstack([np.loadtxt(f) for f in glob(PATH)])
    chewing_data.append(data)
chewing_data=np.vstack(chewing_data)
chewing_label=np.ones((chewing_data.shape[0],1),float)

#process nonchewing features and labels
nonchewing_data=[]
PATH = os.path.join("others from camera", "MFCC1", "*.csv")
data = np.vstack([np.loadtxt(f) for f in glob(PATH)])
nonchewing_data.append(data)
nonchewing_data=np.vstack(nonchewing_data)
nonchewing_label=np.zeros((nonchewing_data.shape[0],1), float)

#merge data and split it into traing set and test set
X=np.vstack((chewing_data, nonchewing_data))
y=np.vstack((chewing_label, nonchewing_label))

np.savetxt('./clean_data/features.txt', X, fmt='%f')
np.savetxt('./clean_data/labels.txt', y, fmt='%d')