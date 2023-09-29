import os
import pickle
import math
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import argparse
import sys
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, target_dir)
from ProcessUtils.feature_extractions import *
import argparse

"""
Params Windows ECG :
--------------------
 - WINDOW_IN_SECONDS: Define a window duration in seconds.
 - NOISE: Specify noise filtering options: 'bp' (band-pass filter), 'time' (noise elimination), 'ens' (ensemble).
 - fs_dicts: Define sampling frequencies for different sensor data.
 - Label_dict: Define a dictionary to map labels to numerical values.
 - sec: Define a variable 'sec' with a value of 12.
 - N: Calculate the number of data points 'N' in a block of 'sec' seconds for BVP sensor data.
 - overlap: Calculate the overlapping length for data blocks, ensuring it's an even number.
"""


WINDOW_IN_SECONDS = 120  # Options: 120 / 180 / 300
NOISE = ['bp_time_ens']

fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
label_dict = {'non-stress': 0, 'stress': 1}
int_to_label = {0: 'non-stress', 1: 'stress'}
sec = 12
N = fs_dict['BVP'] * sec  # one block: 10 sec

overlap = int(np.round(N * 0.02))
overlap = overlap if overlap % 2 == 0 else overlap + 1

class SubjectData:
    def __init__(self, main_path, subject_id):
        self.id = f'S{subject_id}'
        self.subject_key = ["signal", "label", "subject"]
        self.SIGNAL_KEY = ["chest", "wrist"]
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        file_path = os.path.join(main_path, f'{self.id}/{self.id}.pkl')
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file, encoding='latin')
        
        self.labels = self.data['label']

    def get_wrist_data(self):
        wrist_data = self.data['signal']['wrist']
        wrist_data['Resp'] = self.data['signal']['chest']['Resp']
        return wrist_data

    def get_chest_data(self):
        return self.data['signal']['chest']


def extract_ppg_data(e4_data_dict: dict, labels: list, norm_type: str = None) -> pd.DataFrame:
    """
    Extract PPG (Photoplethysmogram) data and labels into a DataFrame, optionally normalizing the data.

    Parameters:
    - e4_data_dict (dict): Dictionary containing PPG data.
    - labels (list): List of labels.
    - norm_type (str, optional): Type of normalization ('std' for standardization, 'minmax' for min-max scaling).

    Returns:
    - df (pd.DataFrame): Processed PPG data in a DataFrame.
    """
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])

    # Replace with the correct sampling frequency for BVP
    fs_bvp = 64 
     # Replace with the correct sampling frequency for labels
    fs_label = 700 
    df.index = [(1 / fs_bvp) * i for i in range(len(df))]
    label_df.index = [(1 / fs_label) * i for i in range(len(label_df))]
    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')
    df = df.join(label_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(axis=0)

    if norm_type == 'std':
        # Standardization (Z-score normalization)
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type == 'minmax':
        # Min-Max scaling
        df = (df - df.min()) / (df.max() - df.min())
    return df

def seperate_data_by_label(df):
    
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)   
    
    return grouped, baseline, stress, amusement


def get_samples(data: pd.DataFrame, label: int,path_root: str ,subject_id ,ma_usage: bool) -> pd.DataFrame:
    """
    Extract samples from PPG data and calculate statistics for each sample window.

    Parameters:
    - data (pd.DataFrame): DataFrame containing PPG data.
    - label (int): Label associated with the samples.
    - ma_usage (bool): Flag indicating whether to use moving average.

    Returns:
    - samples (pd.DataFrame): DataFrame containing the extracted samples and statistics.
    """
    global feat_names
    global WINDOW_IN_SECONDS

    samples = []
    window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS)  # 64 * 60
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)

    winNum = 0
    method = True

    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        # Extract a window of data
        window_data = data[sliding_window_len * i: (sliding_window_len * i) + window_len]
        # Calculate statistics for the window
        wstats = get_rr_interval_spectrogram(ppg_seg=window_data['BVP'].tolist(),
                                              window_length=window_len,
                                              label=label,
                                              path_spectral=path_root,
                                              subject_id=subject_id,
                                              ensemble=ma_usage,
                                              ma_usage=ma_usage)

        winNum += 1

        if not wstats:
            i += 1
            continue

        x = pd.DataFrame(wstats, index=[i])
        samples.append(x)
        i += 1
    return pd.concat(samples).drop_duplicates()

def combine_files(subjects, save_path, subject_feature_path, merged_path):
    df_list = []

    for s in subjects:
        file_path = os.path.join(save_path, subject_feature_path, f'S{s}_feats_4.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df['subject'] = s
            df_list.append(df)

    if not df_list:
        print("No valid files found for merging.")
        return

    df = pd.concat(df_list)
    df['label'] = df.apply(lambda row: 1 if row['0'] == True and row['1'] == True else 0, axis=1)
    df.drop(['0', '1'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    merged_file_path = os.path.join(save_path, merged_path)
    df.to_csv(merged_file_path, index=False)

    counts = df['label'].value_counts()

    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')

def make_patient_data(subject_id, ma_usage, main_path, path_root,subject_feature_path):
    global save_path
    global WINDOW_IN_SECONDS
    
    temp_ths = [1.0,2.0,1.8,1.5] 
    clean_df = pd.read_csv('../ProcessUtils/clean_signal_by_rate.csv',index_col=0)
    cycle = 15
    
    # Make subject data object for Sx
    subject = SubjectData(main_path=main_path, subject_id=subject_id)
    
    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = 'std'

    df = extract_ppg_data(e4_data_dict, subject.labels, norm_type)
    df_BVP = df.BVP
    df_BVP = df_BVP.tolist()


    #여기서 signal preprocessing 
    bp_bvp = butter_bandpassfilter(df_BVP, 0.5, 10, fs_dict['BVP'], order=2) 
    
    if ma_usage:   
        df['BVP'] = bp_bvp
        
    if ma_usage:
        fwd = moving_average(bp_bvp, size=3)
        bwd = moving_average(bp_bvp[::-1], size=3)
        bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
        df['BVP'] = bp_bvp
        
        signal_01_percent = int(len(df_BVP) * 0.001)
        #print(signal_01_percent, int(clean_df.loc[subject_id]['index']))
        clean_signal = df_BVP[int(clean_df.loc[subject_id]['index']):int(clean_df.loc[subject_id]['index'])+signal_01_percent]
        ths = statistic_threshold(clean_signal, fs_dict['BVP'], temp_ths)
        len_before, len_after, time_signal_index = eliminate_noise_in_time(df['BVP'].to_numpy(), fs_dict['BVP'], ths, cycle)
    
        df = df.iloc[time_signal_index,:]
        df = df.reset_index(drop=True)
        
        #plt.figure(figsize=(40,20))
        #plt.plot(df['BVP'][:2000], color = 'b', linewidth=2.5)
    
    
    grouped, baseline, stress, amusement = seperate_data_by_label(df)   
    
    
    baseline_samples = get_samples(baseline, 0,path_root,subject_id,ma_usage)
    stress_samples = get_samples(stress, 1, path_root,subject_id,ma_usage)
    amusement_samples = get_samples(amusement, 0,path_root,subject_id,ma_usage)
    
    print("stress: ",len(stress_samples))
    print("non-stress: ",len(amusement_samples)+len(baseline_samples))
    window_len = len(baseline_samples)+len(stress_samples)+len(amusement_samples)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1) # get dummies로 원핫벡터로 라벨값 나타냄
    
    
    all_samples.to_csv(f'{save_path}{subject_feature_path}/S{subject_id}_feats_4.csv')

    # Does this save any space?
    subject = None
    
    return window_len


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument(
        "--save_path",
        type=str,
        default="SpectralImages/binary/",
        help="Path to save the features (default: SpectralImages/binary/)",
    )

    parser.add_argument(
        "--main_path",
        type=str,
        default="../Data/WESADs/",
        help="Main data path (default: ../Data/WESADs/)",
    )

    return parser

def process_data_for_patient(patient_id, save_path, main_path):
    BP, FREQ, TIME, ENSEMBLE = False, False, False, False

    for n in NOISE:
        if 'bp' in n.split('_'):
            BP = True
        if 'time' in n.split('_'):
            TIME = True
        if 'ens' in n.split('_'):
            ENSEMBLE = True

        subject_feature_path = f'subject_feature_{n}{WINDOW_IN_SECONDS}'
        merged_path = f'data_merged_{n}.csv'

        if not os.path.exists(os.path.join(save_path, subject_feature_path)):
            os.makedirs(os.path.join(save_path, subject_feature_path))

        print(f'Processing data for S{patient_id}...')
        window_len = make_patient_data(patient_id, BP, main_path, save_path,subject_feature_path)
        print('Total window length:', window_len)

    print(f'Processing complete for S{patient_id}.')

if __name__ == "__main__":
    # subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    subject_ids = [2]
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    save_path = args.save_path
    main_path = args.main_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for patient in subject_ids:
        process_data_for_patient(patient, save_path, main_path)

    print('Processing complete for all patients.')