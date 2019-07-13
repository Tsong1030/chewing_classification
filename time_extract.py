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
sys.path.append('./DATA')
from utils import list_files_in_directory, parse_timestamp_tz_aware, read_ELAN, update_ELAN_w_drift, get_timedelta, crop, getDuration
from settings import settings

CROP_ALL_FILES=False #True: Crop all files in VIDEO_FOLDER, False: Only crop those files include chewing

if __name__=="__main__":      
    startDate = settings['START_DATE']
    subj='211'
    VIDEO_FOLDER = os.path.join('VIDEOS',subj, 'VIDEO')
    LABEL_FOLDER = os.path.join('VIDEOS','ANNOTATION', subj, 'CHEWING')
    CHEWING_FOLDER = os.path.join('SOUNDS', 'CHEWING',subj)
    NON_CHEWING_FOLDER= os.path.join('SOUNDS', 'NON_CHEWING',subj)
    os.makedirs(CHEWING_FOLDER, exist_ok=True)
    os.makedirs(NON_CHEWING_FOLDER, exist_ok=True)
    startDateTZ = datetime.combine(startDate[subj], datetime.min.time()).astimezone(settings["TIMEZONE"])

    with open(os.path.join(LABEL_FOLDER, 'sync.yaml')) as f:
        SETTINGS = yaml.load(f)
    
    for episode in SETTINGS:
        print("Processing video: ", episode, "...\n")
    
        # get start time and end time
        syncRelative = SETTINGS[episode]['sync_relative']
        syncAbsolute = SETTINGS[episode]['sync_absolute']
        videoLeadTime = SETTINGS[episode]['video_lead_time']
        syncAbsolute = parse_timestamp_tz_aware(syncAbsolute)
        if len(syncRelative) == 12:
            t = datetime.strptime(syncRelative,"%H:%M:%S.%f")
        elif len(syncRelative) == 8:
            t = datetime.strptime(syncRelative,"%H:%M:%S")
        #datetime object, start time of video with TZ
        startTime = syncAbsolute - timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)\
                    +timedelta(seconds=videoLeadTime/1000)
        
        periods = pd.read_csv(os.path.join(LABEL_FOLDER, episode + '.csv'))
        #heads and tails of each period with TZ
        heads=periods['start']
        tails = periods['end']
        
        #Convert video to tmp wav file for cutting
        infile = os.path.join(VIDEO_FOLDER, episode + '.AVI')
        outfile = episode + '.wav'
        subprocess.call(['ffmpeg', '-y', '-i', infile, outfile])
            
        #cut chewing segments
        print('Cutting chewing segments...\n')
        for i in range(len(periods)):
            #starting offset of the video
            start_offset = get_timedelta(startTime, parse_timestamp_tz_aware(heads[i]))
            print('starting:',start_offset)
            #ending offset of the video
            end_offset=get_timedelta(startTime, parse_timestamp_tz_aware(tails[i]))
            print('ending:', end_offset)       
            targetfile = str(i) + '_' + episode + '.wav'
            print('generating ', targetfile + '.....\n')
            crop(start_offset, end_offset, outfile, os.path.join(CHEWING_FOLDER,targetfile))
        
#        #cut non-chewing segments
        print('Cutting non-chewing segments...\n')
        for i in range(len(periods)+1):
            if i==0:
                start_offset = '00:00:00'
            else:
                start_offset = get_timedelta(startTime, parse_timestamp_tz_aware(tails[i-1]))
            print('starting:',start_offset)
            #ending offset of the video
            if i==len(periods):
                end_offset = getDuration(outfile)
            else:
                end_offset=get_timedelta(startTime, parse_timestamp_tz_aware(heads[i]))
            print('ending:', end_offset)       
            targetfile = str(i) + '_' + episode + '.wav'
            print('generating ', targetfile + '.....\n')
            crop(start_offset, end_offset, outfile, os.path.join(NON_CHEWING_FOLDER,targetfile))
        #remove tmp wave file
        os.remove(outfile)
    
    if CROP_ALL_FILES:
        all_files=list_files_in_directory(VIDEO_FOLDER)
        chewing_files=[ x+".AVI" for x in SETTINGS.keys()]
        for file in all_files:
            if file not in chewing_files:    
                infile = os.path.join(VIDEO_FOLDER, file)
                outfile = '0_' + file[:-4] + '.wav'
                print('generating ', outfile + '.....\n')
                subprocess.call(['ffmpeg', '-y', '-i', infile, os.path.join(NON_CHEWING_FOLDER,outfile)])


