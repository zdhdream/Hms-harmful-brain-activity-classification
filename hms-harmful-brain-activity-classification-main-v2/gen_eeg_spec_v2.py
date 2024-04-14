# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/nartaa/how-to-make-spectrogram-from-eeg/notebook
"""
import librosa
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
data_dir = '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/'
train = pd.read_csv(data_dir+'/train.csv')
print('Train shape', train.shape )
NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((100, 300, 4), dtype='float32')

    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            # FILL NANS
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0

            # COMPUTE PAIR DIFFERENCES
            x = x1 - x2

            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 300,
                                                      n_fft=1024, n_mels=100, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0
    return img


PATH = data_dir+ '/train_eegs/'
DISPLAY = 4
EEG_IDS = train.eeg_id.unique()
all_eegs = {}
import tqdm
for eeg_id in tqdm.tqdm(EEG_IDS):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH}{eeg_id}.parquet')
    all_eegs[eeg_id] = img
print('save..')
# SAVE EEG SPECTROGRAM DICTIONARY
np.save(data_dir+'eeg_specs_v2.npy', all_eegs)