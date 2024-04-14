import os

import numpy as np
import pandas as pd
import librosa
import pywt
import matplotlib.pyplot as plt

NAMES = ['LL', 'LP', 'RP', 'RR']
FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],  # LL
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],  # LP
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],  # RP
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]  # RR

FEATS_V2 = [['Fp1', 'F7', 'F3', 'T3', 'C3'],
            ['T3', 'C3', 'T5', 'P3', 'O1'],
            ['Fp2', 'F4', 'F8', 'C4', 'T4'],
            ['C4', 'T4', 'P4', 'T6', 'O2']]

directory_path = 'EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

PATH = 'data/train_eegs/'
DISPLAY = 4
USE_WAVELET = None


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(eeg_id, parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((256, 512, 4), dtype='float32')

    if display: plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS_V2[k]
        for kk in range(4):
            # COMPUTE PAIR DIFFERENCES AND AVERAGE
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 512,
                                                      n_fft=2048, n_mels=256, fmin=0, fmax=20, win_length=256)

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] = mel_spec_db

        img[:, :, k] /= 4.0
        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect='auto', origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0: offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print();
        print('#' * 25);
        print()

    return img


def main():
    train = pd.read_csv("data/train.csv")
    print('Train shape', train.shape)
    # 一个17089个非重复的脑电图
    EEG_IDS = train.eeg_id.unique()
    all_eegs = {}
    for i, eeg_id in enumerate(EEG_IDS):
        # if (i % 100 == 0) & (i != 0): print(i, ', ', end='')

        img = spectrogram_from_eeg(eeg_id, f"{PATH}{eeg_id}.parquet", i < DISPLAY)

        if i == DISPLAY:
            print(f'Creating and writing {len(EEG_IDS)} spectrograms to disk... ', end='')
        np.save(f'{directory_path}{eeg_id}', img)
        all_eegs[eeg_id] = img
    np.save('data/eeg_spectrograms/eeg_overlap_specs_256x512', all_eegs)


if __name__ == "__main__":
    main()
