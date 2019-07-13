# utils.py(for MD2K)

import os
import sys
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil import parser
from six import string_types
from settings import settings
import pytz
import numpy as np
import subprocess
from sklearn.metrics import *

def get_timedelta(start, end):
    """get timestamp of the video from real time

    Args:
        start:datetime object of video start time
        end: datetime object of certain moment in the video

    Returns:
        string of the timestamp, in format "HH:MM:SS"
        example: '00:04:15'
    """
    delta =  int((end - start).total_seconds())
    mins = delta//60
    secs = delta - mins*60
    return '00:'+str(mins).zfill(2)+':'+str(secs).zfill(2)


def crop(start, end, input, output):
    """crop a .wav file into segments on given time period

    Args:
        start, end:start and end of the period, in format "HH:MM:SS"
        input: origin .wav file to be cropped
        output: generated segment
    """
    command='ffmpeg -y -i '+input+' -ss '+start+' -to '+end+' -c copy '+output
    os.system(command)
     
def getDuration(filename):
    """get duration of a video or wav file

    Args:
        filename: the file to request duration

    Returns:
        string of duration, in format "HH:MM:SS"
        example: '00:20:00'
    """
    result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    for x in result.stdout.readlines():
      if b'Duration' in x:
          return str(x[12:20], 'utf-8')

def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.

    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')

    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 
        
    Return
    ------
        none

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
        argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    with open(logfile, 'a') as out:
        out.write(argAll + '\n')

#################################################################
# 
#   TIME CONVERSION SESSION
# 
#################################################################
def utc_to_local(utc_dt):
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt) # .normalize might be unnecessary


def datetime_to_unixtime(dt):
    '''
    Convert Python datetime object (timezone aware)
    to epoch unix time in millisecond
    '''
    return int(1000 * dt.timestamp())


def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')


def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')


def parse_timestamp_tz_naive(string):
    STARTTIME_FORMAT_WO_CENTURY = '%m/%d/%y %H:%M:%S'
    STARTTIME_FORMAT_W_CENTURY = '%m/%d/%Y %H:%M:%S'
    try:
        dt = datetime.strptime(string, STARTTIME_FORMAT_WO_CENTURY)
    except:
        dt = datetime.strptime(string, STARTTIME_FORMAT_W_CENTURY)

    return dt


def datetime_str_to_unixtime(string):
    return datetime_to_unixtime(parse_timestamp_tz_aware(string))


def parse_timestamp_tz_aware(string):
    return parser.parse(string)


def unixtime_to_datetime(unixtime):
    if len(str(abs(unixtime))) == 13:
        return datetime.utcfromtimestamp(unixtime/1000).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])#.\
            # strftime('%Y-%m-%d %H:%M:%S%z')
    elif len(str(abs(unixtime))) == 10:
        return datetime.utcfromtimestamp(unixtime).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])#.\
            # strftime('%Y-%m-%d %H:%M:%S%z')
    


def check_end_with_timezone(s):# TODO: limit re to -2numbers:2number
    import re
    m = re.search(r'-\d+:\d+$', s)
    if m:
        return True
    else:
        m = re.search(r'-\d{4}', s)
    return True if m else False


def df_to_datetime_tz_aware(in_df, column_list):
    from datetime import datetime, timedelta, date
    from six import string_types
    import numpy as np

    df = in_df.copy()

    for column in column_list:
        if len(df): # if empty df, continue
            d = df[column].iloc[0]
            print(type(d))
            # if type is string 
            if isinstance(d, string_types):#if "import datetime" then "isinstance(x, datetime.date)"
                if check_end_with_timezone(d):
                    # if datetime string end with time zone
                    print('Column '+column+' time zone contained')
                    df[column] = pd.to_datetime(df[column],utc=True).apply(lambda x: x.tz_convert(settings['TIMEZONE']))
                    df[column] = pd.to_datetime(df[column],utc=True).apply(lambda x: x.tz_convert(settings['TIMEZONE']))
                else:
                    # if no time zone contained
                    print('Column '+column+' no time zone contained')
                    df[column] = pd.to_datetime(df[column]).apply(lambda x: x.tz_localize(settings['TIMEZONE']))
            
            # if type is datetime.date
            elif isinstance(d, date):

                if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
                    # if datetime is tz naive
                    print('Column '+column+" time zone naive")
                    df[column] = df[column].apply(lambda x: x.replace(tzinfo=pytz.UTC).astimezone(settings['TIMEZONE']))
                    # df[column] = df[column].apply(lambda x: x.replace(tzinfo=pytz.UTC).astimezone(TIMEZONE))

                else:
                    print(d.tzinfo)
                    # if datetime is tz aware
                    print('Column '+column+" time zone aware")
            # if type is unixtime (13-digit int):
            
            elif isinstance(d, (int, np.int64)) or isinstance(d, (int, np.float)):
                print(settings)          
                df[column] = pd.to_datetime(df[column], unit='ms', utc=True)\
                            .dt.tz_convert(settings["TIMEZONE"])
            
            else:
                print('Cannot recognize the data type.')
        else:
            print('Empty column')

    return df



def string_to_datetime(startDatetime, endDatetime):
    # startDatetime = '06-27-17_11'
    # endDatetime = '06-27-17_15'
    start = datetime.strptime(startDatetime, '%m-%d-%y_%H_%M_%S')
    end = datetime.strptime(endDatetime, '%m-%d-%y_%H_%M_%S')
    return start, end


def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')

def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')



#################################################################
# 
#   PROCESS LABEL FILE SESSION
# 
#################################################################


def read_label_csv_datetime(label_file):
    annot_df = pd.read_csv(label_file)
    annot_df['start'] = pd.to_datetime(annot_df['start'], utc=True)\
                        .dt.tz_convert(settings["TIMEZONE"])
    annot_df['end'] = pd.to_datetime(annot_df['end'], utc=True)\
                        .dt.tz_convert(settings["TIMEZONE"])
    return annot_df


def read_ELAN(path):
    '''
    Read ELAN txt files into a pandas dataframe
    '''
    
    df = pd.read_table(path, header=None)
    df = df.iloc[:,[2,4,-1]]
    df.columns = ['start', 'end', 'label']

    df['start'] = pd.to_timedelta(df['start'])
    df['end']   = pd.to_timedelta(df['end'])
    
    return df


# if drift_time=1s, label should be replaced by label+1s
def update_ELAN_w_drift(ELAN_df, drift_time):
    ELAN_update_df = ELAN_df.copy()
    ELAN_update_df['start'] = ELAN_df['start'] + drift_time
    ELAN_update_df['end'] = ELAN_df['end'] + drift_time

    ELAN_update_df = df_to_datetime_tz_aware(ELAN_update_df, ['start','end'])
    # print(ELAN_update_df)

    # for i in range(len(ELAN_update_df)):
    #     ELAN_update_df['start'].iloc[i] = str(ELAN_update_df['start'].iloc[i])#[7:-3]
    #     ELAN_update_df['end'].iloc[i] = str(ELAN_update_df['end'].iloc[i])#[7:-3]
    return ELAN_update_df


#################################################################
# 
#   FILE/FOLDER OPERATION SESSION
# 
#################################################################


def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def list_folder_in_directory(mypath):
    return [f for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


#################################################################
# 
#   LABEL REPRESENTATION CONVERSION SESSION
# 
#################################################################


def pointwise2headtail(pointwise):
    '''
    return the index of the first non-zero element and the index of the last non-zero element
    
    NOTE: The non-zero element has to be 1.
    >>>pointwise2headtail([0,0,1,1,0])
    >>>[[2 3]]
    '''
    diff = np.concatenate((pointwise[:],np.array([0]))) - np.concatenate((np.array([0]),pointwise[:]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1
    # print(len(ind_tail))
    # print(len(ind_head))
    headtail = np.vstack((ind_head, ind_tail)).T;

    return headtail


def consecutive2headonly(index_arr): 
    '''
    >>>consecutive2headonly([1,2,3,6,7,8,10,11,15,16,17,18,19,20])
    >>>(array([ 1,  6, 10, 15]), [3, 3, 2, 6])
    '''
    
    df = pd.DataFrame(data=index_arr, columns=['orig'])
    df['new'] = 0
    df['new'].iloc[0] = -2
    df['new'].iloc[1:] = index_arr[:-1]
    df['new'] = df['new'] + 1
    df['res'] = df['new'] - df['orig']
    
    head_idx = df[df['res']!=0].index.tolist()
    tail_idx = head_idx[1:]+[len(index_arr)]
    
    len_list = [t-h for (h,t) in zip(head_idx, tail_idx)]

    return df['orig'][df['res']!=0].values, len_list


def map_gt_to_fixed_length(ht, LEVEL):
    '''
    expend [head, tail] to [head_ext, tail_ext] while satisfying 'tail_ext-head_ext+1' is the closest element in the LEVEL list
    >>>ht = [20, 82]
    >>>LEVEL = [40,60,80,100,120]
    >>>print(map_gt_to_fixed_length(ht, LEVEL))
    >>>[11, 90]. (right_append = 9, left_append = 8)


    # when TARGET_LEVEL-l = even, append '(TARGET_LEVEL-l)/2' to left and '(TARGET_LEVEL-l)/2' to right
    # when TARGET_LEVEL-l = odd, append '(TARGET_LEVEL-l)/2 - 1/2' to left and '(TARGET_LEVEL-l)/2 + 1/2' to right

    '''

    head = ht[0]
    tail = ht[1]
    l = tail - head + 1

    LEVEL_arr = np.array(LEVEL)
    right = np.array(np.where(LEVEL_arr>=l))
    # print(right)

    if right.shape[1] == 0:
        lprint('test.txt',datetime.now(),': Longer FG than 6 seconds -  head & tail:', head, ', ', tail)
        return 0
    else:
        target_ind = right[0][0]

        TARGET_LEVEL = LEVEL[target_ind]

        right_append = int((TARGET_LEVEL - l + 1)/2)
        left_append = TARGET_LEVEL - l - right_append
        ht_ext = [head-right_append, tail+left_append]

        return ht_ext


# input para: input_data , intervals_of_interest , timeString 
def gen_feats_names( feat_list , sensor_list, extension ):
    header = []
    
    for feat in feat_list:
        for key in sensor_list:
            one = key + "_" + feat
            header.extend([one])
    header.extend(extension)

    return header


def find_nonmonotonic(x):
    print(x[np.where(np.diff(x) < 0)])
    print(np.where(np.diff(x) < 0))
    print(x[np.where(np.diff(x) < 0)[0]-1])
    print(x[np.where(np.diff(x) < 0)[0]+1])



#################################################################
# 
#   METRCIS SESSION
# 
#################################################################


def calc_multi_cm(y_gt, y_pred):    
    # ct = pd.crosstab(y_gt, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_gt, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    # print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    multi_cm = confusion_matrix(y_gt, y_pred)
    
    accuracy = sum(multi_cm[i,i] for i in range(len(set(y_gt))))/sum(sum(multi_cm[i] for i in range(len(set(y_gt)))))
    recall_all = sum(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt)))) for i in range(len(set(y_gt))))/(len(set(y_gt)))
    precision_all = sum(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt)))) for i in range(len(set(y_gt))))/(len(set(y_gt)))
    fscore_all = sum(2*(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt)))))*(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt)))))/(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt))))+multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt))))) for i in range(len(set(y_gt))))/len(set(y_gt))
    
    result={}

    for i in np.unique(y_gt):

        i_gt = (y_gt==i).astype(int)
        i_pred = (y_pred==i).astype(int)

        cm = confusion_matrix(i_gt, i_pred)

        i_result = {}

        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
        # Precision for Positive = TP/(TP + FP)
        prec_pos = TP/(TP + FP)
        # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
        f1_pos = 2*TP/(TP*2 + FP+ FN)
        # TPR = TP/(TP+FN)
        TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(i_gt))))

        i_result = {'recall': TPR, 'precision': prec_pos, 'f1': f1_pos}

        result[str(int(i))] = i_result

    ave_result = {'accuracy':accuracy, 'recall_all':recall_all, 'precision_all':precision_all, 'fscore_all':fscore_all}
    result['average'] = ave_result

    return result, multi_cm


def wacc_from_CM(cm, n_classes):
    """
    weighted accuracy
    """

    if n_classes == 2:

        TN = cm[0,0]
        TP = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        ratio = float(FP+TN)/float(TP+FN)
        waccuracy = (TP*ratio+TN)/((TP+FN)*ratio+FP+TN)


    elif n_classes == 3:
        waccuracy = (cm[0,0]*(1.0/(cm[0,0]+cm[0,1]+cm[0,2])) + cm[1,1]*(1.0/(cm[1,0]+cm[1,1]+cm[1,2]) + cm[2,2]*(1.0/(cm[2,0]+cm[2,1]+cm[2,2]))))/3.0
        

    
    result = {'waccuracy':waccuracy}

    return result


def metrics_from_CM(multi_cm, n_classes):

    accuracy = sum(multi_cm[i,i] for i in range(n_classes))/sum(sum(multi_cm[i] for i in range(n_classes)))
    recall_all = sum(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)) for i in range(n_classes))/n_classes
    precision_all = sum(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)) for i in range(n_classes))/n_classes
    fscore_all = sum(2*(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)))*(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)))/(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes))+multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes))) for i in range(n_classes))/n_classes
    
    result = {'accuracy':accuracy, 'recall_all':recall_all, 'precision_all':precision_all, 'fscore_all':fscore_all}

    return result



def overall_metrics_from_CM(multi_cm, n_classes):

    accuracy = sum(multi_cm[i,i] for i in range(n_classes))/sum(sum(multi_cm[i] for i in range(n_classes)))
    recall_all = sum(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)) for i in range(n_classes))/n_classes
    precision_all = sum(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)) for i in range(n_classes))/n_classes
    fscore_all = sum(2*(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)))*(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)))/(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes))+multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes))) for i in range(n_classes))/n_classes
    
    result = {'accuracy':accuracy, 'recall_all':recall_all, 'precision_all':precision_all, 'fscore_all':fscore_all}

    return result



def positive_metrics_from_CM(cm):
    # 2 class classification
    # 0 is negative, 1 is positive
    TN = cm[0,0]
    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    ratio = float(FP+TN)/float(TP+FN)
    # print(ratio)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)
    # print(TP*ratio+TN)
    # print((TP+FN)*ratio+FP+TN)
    waccuracy = (TP*ratio+TN)/((TP+FN)*ratio+FP+TN)
    # print(waccuracy)
    recall_pos = TP/(TP+FN)
    precision_pos = TP/(TP+FP)
    fscore_pos = 2*recall_pos*precision_pos/(recall_pos+precision_pos)

    MCC = (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    result = {'waccuracy':waccuracy, 'recall_pos':recall_pos, 'precision_pos':precision_pos, 'fscore_pos':fscore_pos, 'MCC':MCC}

    return result





#################################################################
# 
#   DATA PREPARATION SESSION
# 
#################################################################

def standardize_zscore(data, selected_columns):
    """normalize the whole dataset to make the system cleaner
    only normalize for columns belong to columns list
    Parameters
    ----------
        data:               dataFrame
        selected_columns:   list of columns which will be standardized
    Return
    ------
        dataZ           dataFrame
    """
    dataZ = data.copy()
    for col_header in selected_columns:
        dataZ[col_header] = zscore(dataZ[col_header])

    return dataZ



def tt_split_pseudo_rand(XY, train_ratio, seed):
    # eg: train_ratio = 0.7

    numL = list(range(10))
    random.seed(seed)
    random.shuffle(numL)

    length = len(XY)
    test_enum = numL[0:10-int(10*train_ratio)]
    test_ind = []

    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


# input para: input_data , intervals_of_interest , timeString 
def mark_class_period( df , col, intervals ):
    df[col] = 0
    for ts in intervals:
        a = str(ts[0])
        b = str(ts[1])
        df[col][(df.index >= a) & (df.index < b)] = 1
    return df


def merge_data_multi_label(annot_df, in_data_df, activity_list):
    data_df = in_data_df.copy()
    for act in activity_list:
        annot_act_df = annot_df.loc[annot_df["label"]==act]

        act_st_list = list(annot_act_df.start.tolist())
        act_end_list = list(annot_act_df.end.tolist())
        act_dur = []
        
        for n in range(len(act_st_list)):
            act_dur.append([act_st_list[n],act_end_list[n]])
        data_df = mark_class_period(data_df, act , act_dur)
    return data_df


def truncate_df_index_dt(df, start, end, margin = 30):
    start_dt = start - timedelta(seconds=margin)
    end_dt = end + timedelta(seconds=margin)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


def truncate_df_index_str(df, start, end):

    ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S-%z"

    start_dt = datetime.strptime(start, ABSOLUTE_TIME_FORMAT) - timedelta(seconds=margin)
    end_dt = datetime.strptime(end, ABSOLUTE_TIME_FORMAT) + timedelta(seconds=margin)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df
    

def smoking_zscore(a, axis=0, ddof=0):
    '''
    Customized Z-Score for smoking gesture (keep the mean and unit the std. dev.)
    param:
    ------
    a: numpy array before zscore, N*M (N-time points, M-sensors)

    return:
    b: numpy array after zscore

    test case:
    ----------
    >>>a = np.array(([1,2,2],[2,3,4],[4,5,6],[0,3,5]))
    >>>print(smoking_zscore(a, axis=0))
        [[1.24290745 2.10292133 2.72872234]
         [1.91903085 3.02058427 4.08096915]
         [3.27127766 4.85591014 5.43321596]
         [0.56678404 3.02058427 4.75709255]]
    -------
    '''
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        return ((a - np.expand_dims(mns, axis=axis)) \
                / np.expand_dims(sstd, axis=axis)) \
                + np.expand_dims(mns, axis=axis)
    else:
        return (a - mns) / sstd + mns


def rolling_filter(df, flt_para=4):
    df.accx = df['accx'].rolling(flt_para).mean()
    df.accy = df['accy'].rolling(flt_para).mean()
    df.accz = df['accz'].rolling(flt_para).mean()
    
    # df.rotx = pd.rolling_mean(df.rotx, flt_para)
    # df.roty = pd.rolling_mean(df.roty, flt_para)
    # df.rotz = pd.rolling_mean(df.rotz, flt_para)
    
    # df.pitch_deg = pd.rolling_mean(df.pitch_deg, flt_para)
    # df.roll_deg = pd.rolling_mean(df.roll_deg, flt_para)
    
    df = df.dropna()
    return df


def calc_fft_specfreq(y, freq_set, sampling_rate):

    # freq should be less than one half of the sampling rate
    # the finest granularity of frequency is sampling rate/number of samples

    # (real) freq = n_freq / N * sampling_rate
    # n_freq = N * freq / sampling_rate


    # calculate specific frequencies given freq_set

    # TEST CASE:
    # k = 1 #k = 1,2,3,4
    # arr = np.linspace(0, 2*np.pi, 9)[:-1]
    # arr = np.cos(k*arr)
    # print(np.round(np.abs(scipy.fftpack.fft(arr)), 10))
    # print(calc_fft(arr))
    # 
    # HOW TO CALC THE REAL FREQUENCY FROM FFT OUTPUT?
    # For a sample of size 1024:
    # 1 and -1 are for frequency 1/1024 of sampling rate
    # 2 and -2 are for frequency 2/1024 of sampling rate
    # 3 and -3 are for frequency 3/1024 of sampling rate    

    # Number of samplepoints

    N = y.shape[0]
    (N//sampling_rate)*sampling_rate
    y = y[:]

    if min(freq_set) < sampling_rate/N:
        raise ValueError('Warning: the granularity of frequency should be greater than sampling rate/number of samples!')

    n_freq = [N*freq/sampling_rate for freq in freq_set]

    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])

    return np.round(amp[n_freq],10)


def calc_fft(y):
    # 
    # TEST CASE:
    # k = 1 #k = 1,2,3,4
    # arr = np.linspace(0, 2*np.pi, 9)[:-1]
    # arr = np.cos(k*arr)
    # print(np.round(np.abs(scipy.fftpack.fft(arr)), 10))
    # print(calc_fft(arr))
    # 
    # HOW TO CALC THE REAL FREQUENCY FROM FFT OUTPUT?
    # For a sample of size 1024:
    # 1 and -1 are for frequency 1/1024 of sampling rate
    # 2 and -2 are for frequency 2/1024 of sampling rate
    # 3 and -3 are for frequency 3/1024 of sampling rate
    # 

    # Number of samplepoints
    N = y.shape[0]

    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])

    return np.round(amp,10)


# return fft except the foundamental frequency component
def cal_energy_wo_bf(y):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp[1:])


# return the foundamental/basic frequency component
def cal_energy_bf(y):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(amp[0]*amp[0])


def cal_energy_all(y):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp)


def add_pitch_roll(df):
    accx = df['accx'].as_matrix()
    accy = df['accy'].as_matrix()
    accz = df['accz'].as_matrix()
    
    # strategy 1:
    pitch_deg = 180 * np.arctan (accy/np.sqrt(accx*accx + accz*accz))/pi
    roll_deg = 180 * np.arctan (-accx/accz)/pi
    
    # strategy 2:
#     pitch_deg = 180 * np.arctan (accx/np.sqrt(accy*accy + accz*accz))/pi;
#     roll_deg = 180 * np.arctan (accy/np.sqrt(accx*accx + accz*accz))/pi;
#     yaw_deg = 180 * np.arctan (accz/np.sqrt(accx*accx + accz*accz))/pi;
    
    df["pitch_deg"] = pitch_deg
    df["roll_deg"] = roll_deg
    return df


def continuous_chunk(df, col, max_gap_sec):
    '''
    find continuous chunk in time series data

    return:
    ------
    chunks:
    number_of_chunks:
    '''
    idx_stop = np.where(np.greater(np.diff(df[col]), np.timedelta64(max_gap_sec, 'ms')))[0]
    idx_start = idx_stop + 1
    stop = df[col].iloc[idx_stop].tolist() + [df[col].iloc[-1]] # NOTE: .iloc[] here cannot be skipped
    start = [df[col].iloc[0]] + df[col].iloc[idx_start].tolist()

    chunks = list(zip(start, stop))
    number_of_chunks = len(chunks)
    return chunks, number_of_chunks


def continuous_chunk_unixtime(df, col, max_gap_sec):
    '''
    find continuous chunk in time series data

    return:
    ------
    chunks:
    number_of_chunks:
    '''
    idx_stop = np.where(np.greater(np.diff(df[col]), max_gap_sec))[0]
    idx_start = idx_stop + 1
    stop = df[col].iloc[idx_stop].tolist() + [df[col].iloc[-1]] # NOTE: .iloc[] here cannot be skipped
    start = [df[col].iloc[0]] + df[col].iloc[idx_start].tolist()

    chunks = list(zip(start, stop))
    number_of_chunks = len(chunks)
    return chunks, number_of_chunks


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


if __name__ == '__main__':
    ############################################################
    # test map_gt_to_fixed_length
    # >>> [11, 90]. (right_append = 9, left_append = 8)
    ############################################################
    ht = [20, 82]
    # LEVEL = [40,60,80,100,120]
    LEVEL = [80]
    print(map_gt_to_fixed_length(ht, LEVEL))

    print(datetime_str_to_unixtime('2017-10-03 01:01:24-05:00'))
    print(unixtime_to_datetime(datetime_str_to_unixtime('2017-10-03 01:01:24-05:00')))

    print(datetime_str_to_unixtime('2017-10-03 01:01:24-0500'))
    print(unixtime_to_datetime(datetime_str_to_unixtime('2017-10-03 01:01:24-0500')))

