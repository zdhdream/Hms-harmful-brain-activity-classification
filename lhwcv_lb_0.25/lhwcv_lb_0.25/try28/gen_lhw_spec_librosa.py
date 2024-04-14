# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/nartaa/how-to-make-spectrogram-from-eeg/notebook
"""
import librosa
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
from scipy.signal import butter, lfilter

data_dir = '/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/'
train = pd.read_csv(data_dir+'/train.csv')


FEATS2 = ["Fp1", "T3", "C3", "O1", "F7", "T5", "F3", "P3",
          "Fp2", "C4", "T4", "O2", "F8", "T6", "F4", "P4"]
FEAT2IDX = {x: y for x, y in zip(FEATS2, range(len(FEATS2)))}


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

class Pairs:
    def __init__(self):
        self.pairs1 = [
            ('Fp1', 'T3'),
            ('T3', 'O1'),
            ('F7', 'T5'),
            ('Fp1', 'C3'),
            ('C3', 'O1'),
            ('F3', 'P3'),
            ('Fp2', 'T4'),
            ('T4', 'O2'),
            ('F8', 'T6'),
            ('Fp2', 'C4'),
            ('C4', 'O2'),
            ('F4', 'P4'),
        ]

        self.pairs2 = [
            ('O1', 'T5'),
            ('T5', 'T3'),
            ('T3', 'F7'),
            ('F7', 'Fp1'),
            ('O1', 'P3'),
            ('P3', 'C3'),
            ('C3', 'F3'),
            ('F3', 'Fp1'),
            ('O2', 'T6'),
            ('T6', 'T4'),
            ('T4', 'F8'),
            ('F8', 'Fp2'),
            ('O2', 'P4'),
            ('P4', 'C4'),
            ('C4', 'F4'),
            ('F4', 'Fp2'),
        ]

        self.pairs3 = [
            ('Fp1', 'F7'),
            ('F7', 'T3'),
            ('T3', 'T5'),
            ('T5', 'O1'),
            ('Fp1', 'F3'),
            ('F3', 'C3'),
            ('C3', 'P3'),
            ('P3', 'O1'),
            ('Fp2', 'F8'),
            ('F8', 'T4'),
            ('T4', 'T6'),
            ('T6', 'O2'),
            ('Fp2', 'F4'),
            ('F4', 'C4'),
            ('C4', 'P4'),
            ('P4', 'O2'),
        ]

        self.pairs_map = {}
        self.pairs_map[1] = self.pairs1
        self.pairs_map[2] = self.pairs2
        self.pairs_map[3] = self.pairs3


def _get_eeg_window(file):
    # get center 50 seconds
    eeg = pd.read_parquet(file, columns=FEATS2)
    n_pts = len(eeg)
    EEG_PTS = 10000
    offset = (n_pts - EEG_PTS) // 2
    eeg = eeg.iloc[offset:offset + EEG_PTS]
    eeg_win = np.zeros((EEG_PTS, len(FEATS2)))
    for j, col in enumerate(FEATS2):
        eeg_raw = eeg[col].values.astype("float32")
        # Fill missing values
        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else:
            # All missing
            eeg_raw[:] = 0
        eeg_win[:, j] = eeg_raw
    return eeg_win


def spectrogram_from_eeg(eeg, pairs):
    X = np.zeros((10000, len(pairs)), dtype='float32')
    for i, p in enumerate(pairs):
        X[:, i] = eeg[:, FEAT2IDX[p[0]]] - eeg[:, FEAT2IDX[p[1]]]

    X = np.clip(X, -1024, 1024)
    X = np.nan_to_num(X, nan=0) / 32.0
    X = butter_lowpass_filter(X)
    img = np.zeros((128, 300, len(pairs)), dtype='float32')

    channels = X.shape[1]
    for c in range(channels):
        x = X[:, c]

        mel_spec = librosa.feature.melspectrogram(y=x, sr=200,
                                                  hop_length=len(x) // 300,
                                                  n_fft=1024,
                                                  n_mels=128,
                                                  #fmin=0,
                                                  #fmax=20,
                                                  win_length=128)

        width = (mel_spec.shape[1] // 30) * 30
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]
        img[:, :, c] = mel_spec_db

    return img

PATH = data_dir+ '/train_eegs/'
EEG_IDS = train.eeg_id.unique()


ptype = 2

pairs = Pairs().pairs_map[ptype]

all_eegs = {}
import tqdm
for eeg_id in tqdm.tqdm(EEG_IDS):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    eeg = _get_eeg_window(f'{PATH}{eeg_id}.parquet')
    img = spectrogram_from_eeg(eeg, pairs)
    all_eegs[eeg_id] = img
print('save..')
# SAVE EEG SPECTROGRAM DICTIONARY
np.save(data_dir+f'lhw_specs_ptype{ptype}.npy', all_eegs)