# -*- coding: utf-8 -*-
import random

import torch
import numpy as np
from torch.utils.data import Dataset

import albumentations as albu
from scipy.signal import butter, lfilter
import pickle

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
# FEATS2 = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']

FEATS2 = ["Fp1", "T3", "C3", "O1", "F7", "T5", "F3", "P3",
          "Fp2", "C4", "T4", "O2", "F8", "T6", "F4", "P4"]

FEAT2IDX = {x: y for x, y in zip(FEATS2, range(len(FEATS2)))}


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class DataGenerator(Dataset):
    def __init__(self, data, specs=None, eeg_specs=None, raw_eegs=None, augment=False, mode='train',
                 data_type='eeg',
                 random_common_reverse_signal=0.0,
                 random_common_negative_signal=0.0,
                 random_reverse_signal=0.0,
                 random_negative_signal=0.0,
                 secs=50,
                 ):
        self.data = data
        self.augment = augment
        print('augment: ', augment)
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        fname = '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/aux_infos.pkl'
        self.aux_infos = pickle.load(open(fname, 'rb'))

        self.random_common_reverse_signal = random_common_reverse_signal
        self.random_common_negative_signal = random_common_negative_signal
        self.random_reverse_signal = random_reverse_signal
        self.random_negative_signal = random_negative_signal
        self.secs = secs


        self.bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
        self.rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}

        print('random_common_reverse_signal: ', random_common_reverse_signal)
        print('random_common_negative_signal: ', random_common_negative_signal)
        print('random_reverse_signal: ', random_reverse_signal)
        print('random_negative_signal: ', random_negative_signal)
        print('secs: ', self.secs)
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.data_type == 'hybrid':
            X1, X2, y = self.data_generation(index)

            if self.augment:
                X1 = self.augmentation(X1)

            X1 = torch.from_numpy(X1).permute(2, 0, 1).float()
            X2 = torch.from_numpy(X2).float()
            y = torch.from_numpy(y).float()
            return X1, X2, y

        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        if self.data_type == "raw":
            X = torch.from_numpy(X).float()
        else:
            X = torch.from_numpy(X).permute(2, 0, 1).float()
        y = torch.from_numpy(y).float()
        return X, y

    def data_generation(self, index):
        if self.data_type == 'both':
            X, y = self.generate_all_specs(index)
        elif self.data_type == 'eeg' or self.data_type == 'kaggle':
            X, y = self.generate_specs(index, self.data_type)
        elif self.data_type == 'raw':
            X, y = self.generate_raw(index)
        elif self.data_type == 'hybrid':
            # X1, y = self.generate_all_specs(index)
            X1, y = self.generate_specs(index, data_type='kaggle')
            X2, y = self.generate_raw(index)
            return X1, X2, y

        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        if self.mode == 'test':
            offset = 0
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [spec[offset:offset + 300, k * 100:(k + 1) * 100].T for k in [0, 2, 1, 3]]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # STANDARDIZE PER IMAGE
        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56:100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56:200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56:100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56:200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0 + 56:100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56:200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0 + 56:100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56:200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56:100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56:200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        # EEG
        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56:300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56:400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56:300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56:400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56:300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56:400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56:300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56:400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56:300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56:400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index, data_type='kaggle'):
        # X = np.zeros((512, 512, 3), dtype='float32')
        X = np.zeros((400, 300, 3), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        if self.mode == 'test':
            offset = 0
        else:
            offset = int(row.offset / 2)

        if data_type == 'eeg':
            img = self.eeg_specs[row.eeg_id]
            # img1 = self.eeg_specs[row.eeg_id]
            # img = np.zeros((100, 300, 4), dtype='float32')
            # img[:, :256, :] = img1[:100, :, :]
        elif data_type == 'kaggle':
            spec = self.specs[row.spec_id]
            # offset = len(spec) // 2 - 150

            # item = self.aux_infos[row.eeg_id]
            # offset = int(item['spectrogram_label_offset_seconds_min'] +
            #              item['spectrogram_label_offset_seconds_max']) // 4
            # print('spec shape: ', spec.shape)
            imgs = [spec[offset:offset + 300, k * 100:(k + 1) * 100].T for k in
                    [0, 2, 1, 3]]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)
            # print('img shape: ', img.shape)
            # exit(0)
            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            img = np.nan_to_num(img, nan=0.0)
            img = img.transpose(2, 0, 1).reshape(4 * 100, 300)
            # print('img shape: ', img.shape)
            # exit(0)
        # X[:, :, 0] = img[:200]
        # X[:, :, 1] = img[200:]
        # X[:, :, 2] = img[:200] - img[200:]
        # mn = img.flatten().min()
        # mx = img.flatten().max()
        # ep = 1e-5
        # img = 255 * (img - mn) / (mx - mn + ep)

        X[:, :, 0] = img
        X[:, :, 1] = img
        X[:, :, 2] = img

        # mn = img.flatten().min()
        # mx = img.flatten().max()
        # ep = 1e-5
        # img = 255 * (img - mn) / (mx - mn + ep)
        #
        # X[0 + 56:100 + 56, :256, 0] = img[:, 22:-22, 0]
        # X[100 + 56:200 + 56, :256, 0] = img[:, 22:-22, 2]
        # X[0 + 56:100 + 56, :256, 1] = img[:, 22:-22, 1]
        # X[100 + 56:200 + 56, :256, 1] = img[:, 22:-22, 3]
        # X[0 + 56:100 + 56, :256, 2] = img[:, 22:-22, 2]
        # X[100 + 56:200 + 56, :256, 2] = img[:, 22:-22, 1]
        #
        # X[0 + 56:100 + 56, 256:, 0] = img[:, 22:-22, 0]
        # X[100 + 56:200 + 56, 256:, 0] = img[:, 22:-22, 1]
        # X[0 + 56:100 + 56, 256:, 1] = img[:, 22:-22, 2]
        # X[100 + 56:200 + 56, 256:, 1] = img[:, 22:-22, 3]
        #
        # X[200 + 56:300 + 56, :256, 0] = img[:, 22:-22, 0]
        # X[300 + 56:400 + 56, :256, 0] = img[:, 22:-22, 1]
        # X[200 + 56:300 + 56, :256, 1] = img[:, 22:-22, 2]
        # X[300 + 56:400 + 56, :256, 1] = img[:, 22:-22, 3]
        # X[200 + 56:300 + 56, :256, 2] = img[:, 22:-22, 3]
        # X[300 + 56:400 + 56, :256, 2] = img[:, 22:-22, 2]
        #
        # X[200 + 56:300 + 56, 256:, 0] = img[:, 22:-22, 0]
        # X[300 + 56:400 + 56, 256:, 0] = img[:, 22:-22, 2]
        # X[200 + 56:300 + 56, 256:, 1] = img[:, 22:-22, 1]
        # X[300 + 56:400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def generate_raw(self, index):
        X = np.zeros((10000, 12), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        eeg = self.raw_eegs[row.eeg_id]

        # item = self.aux_infos[row.eeg_id]
        # mi = int(item['eeg_label_offset_seconds_min'])
        # ma = int(item['eeg_label_offset_seconds_max'])
        # # if self.mode == 'train' and random.random() < 0.5 and\
        # #         item['tag'] != 'case4':
        # if self.mode != 'train':
        #     offset = ((mi + ma) / 2) * 200
        # else:
        #     if random.random() < 0.5:
        #         #aug
        #         if random.random() < 0.5 and item['tag'] != 'case4':
        #             offset = random.randint(mi * 200, ma * 200)
        #         else:
        #             #n_pts = len(eeg)
        #             #offset = (n_pts - 10000) // 2
        #             offset = ((mi + ma) / 2) * 200
        #     else:
        #         #same with val, test
        #         offset = ((mi + ma) / 2) * 200
        #
        # offset = int(offset)
        # eeg = eeg[offset:offset + 10000].copy()

        # FEATURE ENGINEER
        # X[:, 0] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['T3']]
        # X[:, 1] = eeg[:, FEAT2IDX['T3']] - eeg[:, FEAT2IDX['O1']]
        #
        # X[:, 2] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['C3']]
        # X[:, 3] = eeg[:, FEAT2IDX['C3']] - eeg[:, FEAT2IDX['O1']]
        #
        # X[:, 4] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['C4']]
        # X[:, 5] = eeg[:, FEAT2IDX['C4']] - eeg[:, FEAT2IDX['O2']]
        #
        # X[:, 6] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['T4']]
        # X[:, 7] = eeg[:, FEAT2IDX['T4']] - eeg[:, FEAT2IDX['O2']]

        X[:, 0] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['T3']]
        X[:, 1] = eeg[:, FEAT2IDX['T3']] - eeg[:, FEAT2IDX['O1']]

        X[:, 2] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['C3']]
        X[:, 3] = eeg[:, FEAT2IDX['C3']] - eeg[:, FEAT2IDX['O1']]

        X[:, 4] = eeg[:, FEAT2IDX["F7"]] - eeg[:, FEAT2IDX["T5"]]
        X[:, 5] = eeg[:, FEAT2IDX["F3"]] - eeg[:, FEAT2IDX["P3"]]
        ##############
        X[:, 6] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['T4']]
        X[:, 7] = eeg[:, FEAT2IDX['T4']] - eeg[:, FEAT2IDX['O2']]

        X[:, 8] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['C4']]
        X[:, 9] = eeg[:, FEAT2IDX['C4']] - eeg[:, FEAT2IDX['O2']]

        X[:, 10] = eeg[:, FEAT2IDX["F8"]] - eeg[:, FEAT2IDX["T6"]]
        X[:, 11] = eeg[:, FEAT2IDX["F4"]] - eeg[:, FEAT2IDX["P4"]]

        # STANDARDIZE
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # if self.mode == "train":
        #     reverse_signal = False
        #     negative_signal = False
        #
        #     if random.uniform(0.0, 1.0) <= self.random_common_reverse_signal:
        #         reverse_signal = True
        #     if random.uniform(0.0, 1.0) <= self.random_common_negative_signal:
        #         negative_signal = True
        #
        #     for i in range(X.shape[1]):
        #         if reverse_signal or random.uniform(0.0, 1.0) <= self.random_reverse_signal:
        #             X[:, i] = np.flip(X[:, i])
        #         if negative_signal or random.uniform(0.0, 1.0) <= self.random_negative_signal:
        #             X[:, i] = -X[:, i]

        # X = butter_bandpass_filter(
        #     X,
        #     self.bandpass_filter["low"],
        #     self.bandpass_filter["high"],
        #     200,
        #     order=self.bandpass_filter["order"],
        # )
        # if self.mode == "train":
        #     for i in range(X.shape[1]):
        #        if random.uniform(0.0, 1.0) <= self.rand_filter["probab"]:
        #           lowcut = random.randint(self.rand_filter["low"], self.rand_filter["high"])
        #           highcut = lowcut + self.rand_filter["band"]
        #           X[:, i] = butter_bandpass_filter(
        #               X[:, i],
        #               lowcut,
        #               highcut,
        #               200,
        #               order=self.rand_filter["order"], )

        # BUTTER LOW-PASS FILTER
        X = self.butter_lowpass_filter(X, order=4)
        # Downsample
        # if self.mode == 'train':
        #     start = np.random.randint(0, 5)  # 随机生成0到4之间的整数
        #     X = X[start::5, :]
        # else:
        #     X = X[::5, :]
        if self.secs != 50:
            lens = self.secs * 200
            if self.mode != 'train':
                X = X[5000-lens//2: 5000 + lens//2]
            else:
                start = random.randint(0, 10000 - lens)
                X = X[start: start + lens]

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def butter_lowpass_filter(self, data, cutoff_freq=30, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, data, axis=0)
        return filtered_data

    def resize(self, img, size):
        composition = albu.Compose([
            albu.Resize(size[0], size[1])
        ])
        return composition(image=img)['image']

    def shift_img(self, img):
        s = random.randint(0, img.shape[1])
        new = np.concatenate([img[:, s:], img[:, :s]], axis=1)
        return new

    def augmentation(self, img):
        mean_v = np.mean(img)
        params4 = {
            "num_masks_x": (2, 4),
            # "num_masks_y": (1, 2),
            # "mask_y_length": (4, 10),
            "mask_x_length": (10, 20),
            "fill_value": mean_v,

        }
        composition = albu.Compose([
            albu.XYMasking(**params4, p=0.4),
            # albu.OneOf(
            #     [
            #         albu.RandomBrightnessContrast(p=1.0, contrast_limit=(-0.2, 0.2), brightness_limit=(-0.1, 0.1)),
            #         #albu.RandomGamma(p=0.5, gamma_limit=(50, 200)),
            #     ],
            #     p=0.5,
            # ),
            # albu.OneOf(
            #     [
            #         albu.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
            #         albu.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0)),
            #         albu.MultiplicativeNoise(
            #             always_apply=False, p=1.0, multiplier=(0.9, 1.1), per_channel=True, elementwise=True
            #         ),
            #     ],
            #     p=0.1,
            # ),
        ])

        # composition = albu.Compose([
        #     albu.HorizontalFlip(p=0.4), # 差 kaggle spec 0.6200
        #     #albu.VerticalFlip(p=0.4), # 差   kaggle spec 0.6255
        #     # no aug kaggle spec 0.6139
        # ])
        img = composition(image=img)['image']
        # if np.random.uniform() < 0.3:
        #     alpha = np.random.uniform(0.8, 1.2)
        #     img = alpha * img
        if np.random.uniform() < 0.3:
            img = self.shift_img(img)

        return img
