import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eeg_from_parquet(parquet_path, FEATS, display=False):
    # EXTRACT MIDDLE 50 SECONDS
    eeg = pd.read_parquet(parquet_path, columns=FEATS)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset:offset + 10_000]  # 提取原始数据的中间10_000行

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # CONVERT TO NUMPY
    data = np.zeros((10_000, len(FEATS)))
    for j, col in enumerate(FEATS):  # 遍历每个特征列

        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)  # 计算x中非NaN值得平均值
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)  # 如果x中存在NaN值，则将NaN值替换为平均值m
        else:
            x[:] = 0  # 如果x全部为NaN值，则将其全部置为0

        data[:, j] = x

        if display:
            if j != 0: offset += x.max()  # 如果不是第一个特征列，则更新偏移量offset
            plt.plot(range(10_000), x - offset, label=col)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split('/')[-1]
        name = name.split('.')[0]
        plt.title(f'EEG {name}', size=16)
        plt.show()

    return data


def main():
    train = pd.read_csv("data/train.csv")
    print(train.shape)
    df = pd.read_parquet("data/train_eegs/1000913311.parquet")
    FEATS = df.columns
    print(f"there are {len(FEATS)} raw eeg features")
    print(list(FEATS))

    print('We will use the following subset of raw features:')
    FEATS = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4','C4', 'P4', 'F8', 'T4', 'T6', 'O2']
    FEAT2IDX = {x: y for x, y in zip(FEATS, range(len(FEATS)))}
    print(list(FEATS))

    all_eegs = {}
    DISPLAY = 4
    EEG_IDS = train.eeg_id.unique()
    PATH = "data/train_eegs/"

    CREATE_EEGS = True
    TRAIN_MODEL = True

    for i, eeg_id in enumerate(EEG_IDS):
        if (i % 100 == 0) & (i != 0): print(i, ', ', end='')

        # SAVE EEG TO PYTHON DICTIONARY OF NUMPY ARRAYS
        data = eeg_from_parquet(f'{PATH}{eeg_id}.parquet', FEATS, display=i < DISPLAY)
        all_eegs[eeg_id] = data

        if i == DISPLAY:
            if CREATE_EEGS:
                print(f'Processing {train.eeg_id.nunique()} eeg parquets... ', end='')
            else:
                print(f'Reading {len(EEG_IDS)} eeg NumPys from disk.')
                break

    if CREATE_EEGS:
        np.save('eegs', all_eegs)
    else:
        all_eegs = np.load('data/brain-eegs/eegs_16.npy', allow_pickle=True).item()


if __name__ == "__main__":
    main()
