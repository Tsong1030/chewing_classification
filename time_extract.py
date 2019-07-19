import os
import sys
import yaml
import csv
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dateutil
from dateutil import parser
import subprocess
sys.path.append('../..')
from utils import list_files_in_directory, parse_timestamp_tz_aware, read_ELAN, update_ELAN_w_drift
from settings import settings


startDate = settings['START_DATE']
subj='211'
#VIDEO_FOLDER = os.path.join(subj, 'VIDEO')
folder_path = os.path.join('/Users/fanfeimeng/Desktop/Research/Raw Data/disk data/ANNOTATION', subj, 'CHEWING')
startDateTZ = datetime.combine(startDate[subj], datetime.min.time()).\
        astimezone(settings["TIMEZONE"])
ELANAnnotDfConcat = []
setting_path = os.path.join(folder_path, 'sync.yaml')
#time_episode_file = 'episode'
#time_order_file = 'order'
#start_label = 'start_time_label'
#end_label = 'end_time_label'
#dataheader = ['episode','order','start_time_label', 'end_time_label']
#data_original = pd.read_csv(filepath, sep=',', header=0, names=dataheader, skip_blank_lines=True)
info_docu = pd.read_csv('info_docu.csv')
#print('info_docu.csv:',info_docu)
#info_docu.loc['start'] = '0'
print('info_docu.csv:',info_docu)
#exit()

with open(setting_path) as f:
    SETTINGS = yaml.load(f)

ELANAnnotDfConcat = []

for episode in SETTINGS:
    print(episode)

    # get start time and end time
    syncRelative = SETTINGS[episode]['sync_relative']
    syncAbsolute = SETTINGS[episode]['sync_absolute']
    videoLeadTime = SETTINGS[episode]['video_lead_time']

    syncAbsolute = parse_timestamp_tz_aware(syncAbsolute)
    if len(syncRelative) == 12:
        t = datetime.strptime(syncRelative,"%H:%M:%S.%f")
    elif len(syncRelative) == 8:
        t = datetime.strptime(syncRelative,"%H:%M:%S")

    startTime = syncAbsolute - timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)\
                +timedelta(seconds=videoLeadTime/1000)
    print("startTime:",startTime)
    AAA = pd.read_csv(os.path.join(folder_path, episode + '.csv'))
    print ('A:',AAA)
    s=0
    B=AAA['start']
    print('start time for the file:',AAA['end'])
    length = len(B)
    print ('B:',B)
    C = AAA['end']
    print('end time for the file:',C)
    B = pd.to_datetime(B)
    C = pd.to_datetime(C)
    startTime = time.mktime(startTime.timetuple())
    print('startTime for video:',startTime)
    while length > s:
        print('B.iloc[s]:',B.iloc[s])
        #B.iloc[s] is the starting time of label
        b_tmp = B.iloc[s]
        b_tmp = time.mktime(b_tmp.timetuple())
        print('b_tmp:', b_tmp)
        delta1 =  b_tmp - startTime
        print('delta1:', delta1)
        min1 = int(delta1//60)
        sec1 = int(delta1 - min1*60)
        print('sec1:',sec1)
        if min1<10:
            min1 = str(min1)
            min1='0'+min1
        else:
            min1=str(min1)
        if sec1<10:
            sec1=str(sec1)
            sec1='0'+sec1
        else:
           sec1=str(sec1)
        print('min1',min1)
        hour1 = '00'
        starting = hour1+':'+min1+':'+sec1
        print('starting:',starting)
        #C.iloc[s] is the ending time of label
        c_tmp = C.iloc[s]
        print('C:',C)
        c_tmp=time.mktime(c_tmp.timetuple())
        print('c_tmp:', c_tmp)
        delta2 = c_tmp-startTime
        print('delta2:', delta2)
        min2 = int(delta2 // 60)
        sec2 = int(delta2 - min2 * 60)
        print('sec2:', sec2)
        if min2 < 10:
            min2 = str(min2)
            min2 = '0' + min2
        else:
            min2 = str(min2)
        if sec2<10:
            sec2=str(sec2)
            sec2='0'+sec2
        else:
           sec2=str(sec2)
        print('min2', min2)
        hour2 = '00'
        ending = hour2 + ':' + min2 + ':' + sec2
        print('ending:', ending)
        #生成两列的csv文件，第一列是文件名episode，第二列是每个文件里面的每行label的顺序标记
        tttt=0
        #print('type of b_tmp；',type(b_tmp))
        #b_tmp = b_tmp.astype('object')
        #info_docu.loc['start',s] = b_tmp
        info_docu.loc[s, 'start'] = B.iloc[s]
        print(info_docu.columns)
        cc = info_docu.loc[:,'start']
        #cc = cc.append(pd.DataFrame({'start':[b_tmp]}))
        print('start_label',cc)
        #print('end_label',end_label)
        #while length > tttt:
        #    time_episode_file = np.vstack((time_episode_file,episode))
        #    time_order_file = np.vstack((time_order_file,s))
        #    time_label_file = np.hstack((time_episode_file,time_order_file))
        #    tttt = tttt+1
        s = s + 1
#csv_timelabel = episode + '.csv'
#np.savetxt(csv_timelabel, time_label_file)

