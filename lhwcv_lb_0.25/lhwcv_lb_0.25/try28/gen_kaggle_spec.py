# -*- coding: utf-8 -*-
import os
import tqdm
import pandas as pd
import numpy as np
READ_SPEC_FILES = False

# READ ALL SPECTROGRAMS
PATH = '/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

spectrograms = {}
for  f in tqdm.tqdm(files):
    tmp = pd.read_parquet(f'{PATH}{f}')
    name = int(f.split('.')[0])
    spectrograms[name] = tmp.iloc[:, 1:].values
np.save('/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/kaggle_specs.npy', spectrograms)