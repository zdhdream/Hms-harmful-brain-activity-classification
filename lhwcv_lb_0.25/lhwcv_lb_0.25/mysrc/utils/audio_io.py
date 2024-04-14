# -*- coding: utf-8 -*-
# Copyright  2022 CZSL

import numpy as np
import soundfile
import librosa
import io

def wav_read(filename, tgt_fs=None):
    y, sr = soundfile.read(filename, dtype='float32')
    if tgt_fs is not None:
        if sr != tgt_fs:
            y = librosa.resample(y, orig_sr=sr, target_sr=tgt_fs)
            sr = tgt_fs
    return y, sr

def wav_write(data, fs, filename):
    max_value_int16 = (1 << 15) - 1
    data_save = data * max_value_int16
    soundfile.write(filename, data_save.astype(np.int16), fs, subtype='PCM_16',
             format='WAV')

def read_pcm_16_float(filename, channels=1):
    with open(filename,'rb') as f:
        #max_value_int16 = (1 << 15) - 1
        pcm_data = np.fromfile(f, dtype=np.int16)
        if channels!=1:
            #data = data[: size//channels * channels]
            pcm_data = pcm_data.reshape(-1, channels)
        data = np.array(pcm_data) / 32768

        return data
