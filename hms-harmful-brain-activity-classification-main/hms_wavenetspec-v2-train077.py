import sys
import os
import gc
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda import amp

import timm
from timm.models.layers import get_act_layer
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transforms import *

from kaggle_kl_div import score

from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

from triplet_attention import TripletAttention

from scipy.signal import butter, lfilter

from architectures import *
from models1d_pytorch.wavegram import WaveNetSpectrogram, CNNSpectrogram, ResNetSpectrogram

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA = "data"
TRAIN_SPEC = f"{DATA}/train_spectrograms"
TEST_SPEC = f"{DATA}/test_spectrograms"

TMP = f"{DATA}/tmp"
TRAIN_SPEC_SPLIT = f"{TMP}/train_spectrograms_split"
TEST_SPEC_SPLIT = f"{TMP}/test_spectrograms_split"
if not os.path.exists(TMP):
    os.mkdir(TMP)
if not os.path.exists(TRAIN_SPEC_SPLIT):
    os.mkdir(TRAIN_SPEC_SPLIT)
if not os.path.exists(TEST_SPEC_SPLIT):
    os.mkdir(TEST_SPEC_SPLIT)

RANDOM_SEED = 8620
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
# FOLDS = [0, 1]
N_FOLDS = len(FOLDS)

# N_WORKERS = os.cpu_count() // 2
N_WORKERS = 0

# Wave
eeg_features = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
# eeg_features = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}


class CFG:
    # base parameters
    exp_num = 4
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda:1"
    train_batch_size = 16
    val_batch_size = 32
    IMG_SIZE = [512, 512]
    type = "wave"
    # backbone && model parameters
    model_name = "tf_efficientnet_b2"  # tf_efficientnet_b5 tf_efficientnet_b2_ap tf_efficientnetv2_s.in1k
    max_len = 512
    in_channels = 3
    head_dropout = 0.2
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    pooling = 'gem'
    drop_rate = None
    # GeM
    gemp_p = 4.0
    gemp_learn_p = False
    overwrite_gem_p = None
    # triplet attention
    triplet_attention = True
    triplet_kernel_size = 7
    # act_layer
    act_layer = "swish"  # relu / swish or silu /mish /leaky_relu / prelu...
    # optimizer && scheduler parameters
    lr = 1e-3
    lr_ratio = 5
    min_lr = 4e-5 / 2
    warmupstep = 0
    encoder_lr = 0.0005
    decoder_lr = 0.0005
    weight_decay = 1e-6
    eps = 1.0e-06
    betas = [0.9, 0.999]
    # training parameters
    epochs = 15
    epochs_2 = 8
    es_patience = 4
    # augmentation parameters
    mixup_out_prob = 0.5
    mixup_in_prob1 = 0.5
    mixup_in_prob2 = 0.5
    mixup_alpha_in = 5.0
    mixup_alpha_out = 5.0
    grad_acc = 1
    # wave parameters v1
    wave_model_name = "tf_efficientnetv2_s.in1k"
    timm_params = dict(
        drop_rate=0.2,
        drop_path_rate=0.2,
    )
    spec_params = dict(
        in_channels=1,
        base_filters=128,
        wave_layers=(10, 6, 2),
        kernel_size=3,
    )
    resize_img = None
    custom_classifier = "gem"
    upsample = "bicubic"
    transforms = dict(
        train=Compose([
            # DWTDenoise(p=0.5),
            FlipWave(p=0.5),
        ])
    )


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):

    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())

    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []

        return final_metric


def get_transforms(CFG):
    train_transform = A.Compose([
        # A.VerticalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        # A.Resize(p=1.0, height=CFG.height, width=CFG.width),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length: length + 1]
    h = np.exp(-(x ** 2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    label = np.convolve(label[:, ], gaussian_kernel(offset, sigma), mode="same")
    return label


class HMSHBASpecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
            specs,
            eeg_specs,
            transform: A.Compose,
            phase: str
    ):
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5
        self.cfg = CFG

    def __len__(self):
        return len(self.data)

    def smooth_labels(self, labels, factor=0.001):
        labels *= (1 - factor)
        labels += (factor / 6)
        return labels

    def make_Xy(self, new_ind):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((512, 512), dtype="float32")

        row = self.data.iloc[new_ind]
        r = int((row['min'] + row['max']) // 4)

        # for k in range(1):
        # extract transform spectrogram
        # img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T  # (100, 300)
        img = self.specs[row.spec_id][:, :].T  # (256, 512 or ???)
        # print(row.spec_id, img.shape)
        ch = img.shape[1] // 2
        if ch >= 256:
            img = self.specs[row.spec_id][ch - 256:ch + 256, :].T  # (256, 512)
        else:
            img = self.specs[row.spec_id][:, :].T  # (256, ???)

        # print(row.spec_id, img.shape)
        h, w = img.shape[:2]

        # log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        # crop to 256 time steps
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (256, 512, 4)

        X[0:256, :, 1] = img[:, :, 0]  # (256, 512, 5)
        X[256:512, :, 1] = img[:, :, 1]  # (256, 512, 5)
        X[0:256, :, 2] = img[:, :, 2]  # (256, 512, 5)
        X[256:512, :, 2] = img[:, :, 3]  # (256, 512, 5)

        if self.phase == 'train':
            X = self.spec_mask(X)

            if torch.rand(1) > self.aug_prob:
                X = self.shift_img(X)

            # if torch.rand(1) > self.aug_prob:
            #     X = self.lower_upper_freq(X)

        X = self._apply_transform(X)

        if self.phase == 'train':
            y[:] = self.smooth_labels(row[CLASSES])
            # y[:] = gaussian_label(row[CLASSES], offset=2, sigma=1)
        else:
            y[:] = row[CLASSES]

        return X, y

    def __getitem__(self, index: int):

        X1, y1 = self.make_Xy(index)

        if torch.rand(1) > 0.5 and self.phase == 'train':
            index2 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X2, y2 = self.make_Xy(index2)
        else:
            X2, y2 = X1, y1

        if torch.rand(1) > 0.5 and self.phase == 'train':
            index3 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X3, y3 = self.make_Xy(index3)

        else:
            X3, y3 = X1, y1

        y = (y1 + y2 + y3) / 3

        return {"data1": X1, "data2": X2, "data3": X3, "target": y}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img

    def shift_img(self, img):
        s = torch.randint(0, CFG.IMG_SIZE[1], (1,))[0]
        new = np.concatenate([img[:, s:], img[:, :s]], axis=1)
        return new

    def spec_mask(self, img, max_it=4):
        count = 0
        new = img
        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[0] - CFG.IMG_SIZE[0] // 16, (1,))[0]
            h = torch.randint(CFG.IMG_SIZE[0] // 32, CFG.IMG_SIZE[0] // 16, (1,))[0]
            new[s:s + h] *= 0
            count += 1

        count = 0

        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[1] - CFG.IMG_SIZE[1] // 16, (1,))[0]
            w = torch.randint(CFG.IMG_SIZE[1] // 32, CFG.IMG_SIZE[1] // 16, (1,))[0]
            new[:, s:s + w] *= 0
            count += 1
        return new

    def lower_upper_freq(self, images):
        r = torch.randint(256, 512, size=(1,))[0].item()
        x = (torch.rand(size=(1,))[0] / 2).item()
        pink_noise = (
            np.array(
                [
                    np.concatenate(
                        (
                            1 - np.arange(r) * x / r,
                            np.zeros(512 - r) - x + 1,
                        ),
                        axis=0
                    )
                ]
                , dtype=np.float32).T
        )
        images = images * pink_noise
        return images


def butter_lowpass_filter(data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


class HMSWaveSpecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 eegs,
                 transforms=None,
                 downsample: int = 5,
                 phase: str = "train",
                 ):
        self.data = data
        self.eegs = eegs
        self.transforms = transforms
        self.downsample = downsample
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def smooth_labels(self, labels, factor=0.001):
        labels *= (1 - factor)
        labels += (factor / 6)
        return labels

    def __data_generation(self, index):
        row = self.data.iloc[index]
        X = np.zeros((10_000, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        data = self.eegs[row.eeg_id]

        X[:, 0] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['T3']]
        X[:, 1] = data[:, feature_to_index['T3']] - data[:, feature_to_index['O1']]

        X[:, 2] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['C3']]
        X[:, 3] = data[:, feature_to_index['C3']] - data[:, feature_to_index['O1']]

        X[:, 4] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['C4']]
        X[:, 5] = data[:, feature_to_index['C4']] - data[:, feature_to_index['O2']]

        X[:, 6] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['T4']]
        X[:, 7] = data[:, feature_to_index['T4']] - data[:, feature_to_index['O2']]

        # Standarize
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # Butter low-pass filter
        X = butter_lowpass_filter(X).astype(np.float32)

        if self.phase == 'train':
            y[:] = self.smooth_labels(row[CLASSES])
            # y[:] = gaussian_label(row[CLASSES], offset=2, sigma=1)
        else:
            y[:] = row[CLASSES]

        return X, y

    def __getitem__(self, index: int):
        X, y = self.__data_generation(index)
        X = X[::self.downsample, :]  # (2000, 8)
        X = X.reshape(-1, )
        X = X[None, :]
        return {"data": X, "target": y}


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data, scalos=None, specs=None, eeg_specs=None, spec_aug=None,
                 augmentations=None, phase=False, aug_prob=0.5, smooth_factor=1e-3):
        self.data = data
        self.eeg_scalos = scalos
        self.specs = specs  # Spectrograms for each ID
        self.eeg_specs = eeg_specs  # EEG spectrograms for each ID
        self.augmentations = augmentations
        self.phase = phase
        self.spec_aug = spec_aug
        self.aug_prob = aug_prob
        self.smooth_factor = smooth_factor

    def __len__(self):
        return len(self.data)

    def smooth_labels(self, labels, factor=1e-3):
        labels *= (1 - factor)
        labels += (factor / 6)
        return labels

    def __getitem__(self, index):
        row1 = self.data.iloc[index]
        row2 = self.data.iloc[index]
        contra_tgt = 1
        if self.phase == 'train' and 0.5 > torch.rand(1):
            ind2 = torch.randint(0, self.__len__(), (1,)).tolist()[0]
            row2 = self.data.iloc[ind2]
            if index != ind2: contra_tgt = 0

        # Processing spectrogram data
        spec = self.__spec_data_generation(row1)
        scalo = self.__scalo_data_generation(row2)
        if self.spec_aug is not None:
            # spec = self.__transform(spec)
            spec = self.spec_aug(image=spec.copy())['image']
            scalo = self.spec_aug(image=scalo.copy())['image']

        if self.phase != 'test':
            label1 = row1[CLASSES]  # Assuming 'TARGETS' is defined somewhere as the label column name
            label2 = row2[CLASSES]
            label12 = (label1 + label2) / 2.0
            if self.phase == 'train':
                label1 = self.smooth_labels(label1, self.smooth_factor)
                label2 = self.smooth_labels(label2, self.smooth_factor)
                label12 = self.smooth_labels(label12, self.smooth_factor)
            label1 = torch.tensor(label1).float()
            label2 = torch.tensor(label2).float()
            label12 = torch.tensor(label12).float()
            return {'scalo': scalo, 'spec': spec, 'label12': label12, 'label1': label1, 'label2': label2,
                    'contra_tgt': contra_tgt}
        else:
            return {'scalo': scalo, 'spec': spec}

    def __spec_data_generation(self, row):
        """
        Generates data containing batch_size samples. This method directly
        uses class attributes for spectrograms and EEG spectrograms.
        """
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((512, 512), dtype="float32")
        if self.phase != 'test':
            # Assuming your DataFrame has a column that combines min and max values for slicing
            r = int((row['min'] + row['max']) // 4)
        else:
            r = 0  # Adjust as necessary for test mode

        # img = self.specs[row.spec_id][:, :].T  # (256, ???)

        ch = img.shape[1] // 2
        if ch >= 256:
            img = self.specs[row.spec_id][:, :].T  # (256, 512)
            center = img.shape[1] // 2
            img = img[:, center - 256:center + 256]
        else:
            img = self.specs[row.spec_id][:, :].T  # (256, ???)

        # print(row.spec_id, img.shape)
        h, w = img.shape[:2]

        # log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        # crop to 256 time steps
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (256, 512, 4)

        X[0:256, :, 1] = img[:, :, 0]  # (256, 512, 5)
        X[256:512, :, 1] = img[:, :, 1]  # (256, 512, 5)
        X[0:256, :, 2] = img[:, :, 2]  # (256, 512, 5)
        X[256:512, :, 2] = img[:, :, 3]  # (256, 512, 5)

        if self.phase == 'train':
            X = self.spec_mask(X)
            if torch.rand(1) < self.aug_prob:
                X = self.shift_img(X)

            if torch.rand(1) < 0.5:
                X = X[:, ::-1]

        return X

    def __scalo_data_generation(self, row):
        """
        Generates data containing batch_size samples. This method directly
        uses class attributes for spectrograms and EEG spectrograms.
        """
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((512, 512), dtype="float32")
        if self.phase != 'test':
            # Assuming your DataFrame has a column that combines min and max values for slicing
            r = int((row['min'] + row['max']) // 4)
        else:
            r = 0  # Adjust as necessary for test mode

        # img = self.specs[row.spec_id][:, :].T  # (256, ???)

        ch = img.shape[1] // 2
        if ch >= 256:
            img = self.specs[row.spec_id][:, :].T  # (256, 512)
            center = img.shape[1] // 2
            img = img[:, center - 256:center + 256]
        else:
            img = self.specs[row.spec_id][:, :].T  # (256, ???)

        # print(row.spec_id, img.shape)
        h, w = img.shape[:2]

        # log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        # crop to 256 time steps
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_scalos[row.eeg_id]  # (256, 512, 4)

        X[0:256, :, 1] = img[:, :, 0]  # (256, 512, 5)
        X[256:512, :, 1] = img[:, :, 1]  # (256, 512, 5)
        X[0:256, :, 2] = img[:, :, 2]  # (256, 512, 5)
        X[256:512, :, 2] = img[:, :, 3]  # (256, 512, 5)

        if self.phase == 'train':
            X = self.spec_mask(X)
            if torch.rand(1) < self.aug_prob:
                X = self.shift_img(X)

            if torch.rand(1) < 0.5:
                X = X[:, ::-1]

        return X

    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),

        ])
        return transforms(image=img)['image']

    def shift_img(self, img):
        s = torch.randint(0, CFG.IMG_SIZE[1], (1,))[0]
        new = np.concatenate([img[:, s:], img[:, :s]], axis=1)
        return new

    def spec_mask(self, img, max_it=6):
        count = 0
        new = img
        while count < max_it and torch.rand(1) < self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[0] - CFG.IMG_SIZE[0] // 16, (1,))[0]
            h = torch.randint(CFG.IMG_SIZE[0] // 32, CFG.IMG_SIZE[0] // 16, (1,))[0]
            new[s:s + h] *= 0
            count += 1

        count = 0

        while count < max_it and torch.rand(1) < self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[1] - CFG.IMG_SIZE[1] // 16, (1,))[0]
            w = torch.randint(CFG.IMG_SIZE[1] // 32, CFG.IMG_SIZE[1] // 16, (1,))[0]
            new[:, s:s + w] *= 0
            count += 1
        return new


def get_optimizer(model):
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    return Adam(model.parameters(), **optimizer_params)


def get_scheduler(optimizer, min_lr, epochs, warmupstep=0, warmup_lr_init=1e-5):
    # document:https://timm.fast.ai/SGDR
    return CosineLRScheduler(
        optimizer,
        t_initial=60,
        lr_min=min_lr,
        warmup_t=warmupstep,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=True,
        k_decay=1
    )


def train_wave_one_fold(cfg, fold, train_index, val_index, train_all, brain_eegs, output_path):
    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(cfg)

    train_dataset = HMSWaveSpecDataset(train_all.iloc[train_index],
                                       eegs=brain_eegs,
                                       transforms=cfg.transforms["train"],
                                       downsample=5,
                                       phase='train')

    val_dataset = HMSWaveSpecDataset(train_all.iloc[val_index],
                                     eegs=brain_eegs,
                                     downsample=5,
                                     phase='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, num_workers=N_WORKERS, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

    model = SpectroCNN(
        model_name=cfg.wave_model_name,
        pretrained=True,
        num_classes=6,
        timm_params=cfg.timm_params,
        spectrogram=WaveNetSpectrogram,
        spec_params=cfg.spec_params,
        resize_img=None,
        custom_classifier=cfg.custom_classifier,
        upsample=cfg.upsample,
    )

    model.to(device)

    model_ema = ModelEmaV2(model, decay=0.99)

    # optimizer = get_optimizer(
    #     model,
    #     cfg.lr,
    #     cfg.lr_ratio
    # )
    optimizer = get_optimizer(model)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)

    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()

    loss_func_val_ema = KLDivLossWithLogitsForVal()

    use_amp = cfg.enable_amp
    scaler = amp.GradScaler(enabled=use_amp, init_scale=8192.0)

    best_val_loss = 1
    best_val_loss_ema = 1
    best_epoch = 0
    best_epoch_ema = 0
    train_loss = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time()
        model.train()
        with tqdm(train_loader, leave=True) as pbar:
            for idx, batch in enumerate(pbar):
                batch = to_device(batch, device)
                x, t = batch["data"], batch["target"]

                optimizer.zero_grad()
                with amp.autocast(use_amp, dtype=torch.float16):
                    y = model(x)
                    loss = loss_func(y, t)

                train_loss += loss.item()
                if cfg.grad_acc > 1:
                    loss = loss / cfg.grad_acc
                scaler.scale(loss).backward()

                if (idx + 1) % cfg.grad_acc == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                        # scheduler.step(epoch + idx / len(train_loader))

                if not math.isfinite(loss):
                    print(f"Loss is {loss}, stopping training")
                    sys.exit(1)

                pbar.set_postfix(
                    OrderedDict(
                        loss=f'{loss.item() * cfg.grad_acc:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )

                model_ema.update(model)

        train_loss /= len(train_loader)

        model.eval()
        for batch in tqdm(val_loader):
            x, t = batch["data"], batch["target"]

            x = to_device(x, device)

            with torch.no_grad(), amp.autocast(use_amp, dtype=torch.float16):
                y = model(x)
                y_ema = model_ema.module(x)
            y = y.detach().cpu().to(torch.float32)
            y_ema = y_ema.detach().cpu().to(torch.float32)

            loss_func_val(y, t)
            loss_func_val_ema(y_ema, t)

        val_loss = loss_func_val.compute()
        val_loss_ema = loss_func_val_ema.compute()

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            if CFG.device != 'cuda:1':
                model.to('cuda:1')
            torch.save(model.state_dict(), str(f"{output_path}/snapshot_epoch_{epoch}.pth"))
            if CFG.device != 'cuda:1':
                model.to(CFG.device)

        if val_loss_ema < best_val_loss_ema:
            best_epoch_ema = epoch
            best_val_loss_ema = val_loss_ema
            if CFG.device != 'cuda:1':
                model_ema.to('cuda:1')
            torch.save(model_ema.module.state_dict(), str(f"{output_path}/snapshot_epoch_{epoch}_ema.pth"))
            if CFG.device != 'cuda:1':
                model_ema.to(CFG.device)

        elapsed_time = time() - epoch_start
        print(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, val loss ema: {val_loss_ema: .6f}, elapsed_time: {elapsed_time: .3f}sec")

        if epoch - best_epoch > CFG.es_patience and epoch - best_epoch_ema > CFG.es_patience:
            print("Early Stopping!")
            break

        train_loss = 0
    return fold, best_epoch, best_val_loss, best_epoch_ema, best_val_loss_ema


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    # model.half()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            # x = x.half()
            y = model(x)
            pred_list.append(y.float().softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


def main():
    df = pd.read_csv(f"{DATA}/train.csv")

    train = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
    train = train.rename(columns={"spectrogram_id": "spec_id", "spectrogram_label_offset_seconds": "min"})

    tmp = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds': 'max'})
    train['max'] = tmp

    tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp

    tmp = df.groupby('eeg_id')[CLASSES].agg('sum')
    for t in CLASSES:
        train[t] = tmp[t].values

    y_data = train[CLASSES].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[CLASSES] = y_data

    tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train['target'] = tmp

    tmp = df.groupby('eeg_id')[['label_id']].agg('first')
    train['label_id'] = tmp

    labels = train[CLASSES].values + 1e-5
    train['kl'] = torch.nn.functional.kl_div(
        torch.log(torch.tensor(labels)),
        torch.tensor([1 / 6] * 6),
        reduction='none'
    ).sum(dim=1).numpy()

    train = train.reset_index()

    brain_eegs = np.load(f"{DATA}/brain-eegs/eegs.npy", allow_pickle=True).item()

    score_list = []
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):
        output_path = Path(f"{CFG.exp_num:03d}_fold{i}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{i}]")
        score_list.append(
            train_wave_one_fold(CFG, i, train_index, valid_index, train, brain_eegs, output_path))
    print(score_list)

    out_final_path = Path(f"res_{CFG.exp_num:03d}")
    out_final_path.mkdir(exist_ok=True)

    for (fold_id, best_epoch, _, best_epoch_ema, _) in score_list:
        exp_dir_path = Path(f"{CFG.exp_num:03d}_fold{fold_id}")
        best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
        copy_to = f"{out_final_path}/best_model_fold{fold_id}.pth"
        shutil.copy(best_model_path, copy_to)

        best_model_path_ema = exp_dir_path / f"snapshot_epoch_{best_epoch_ema}_ema.pth"
        copy_to = f"{out_final_path}/best_model_fold{fold_id}_ema.pth"
        shutil.copy(best_model_path_ema, copy_to)

        for p in exp_dir_path.glob("*.pth"):
            p.unlink()

    # Inference Out of Fold

    all_oof = []
    all_oof_ema = []
    all_true = []
    ids = []

    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):
        print(f"\n[fold {i}]")
        device = torch.device(CFG.device)

        # get transform
        _, val_transform = get_transforms(CFG)

        # get_dataloader

        val_dataset = HMSWaveSpecDataset(train.iloc[valid_index],
                                         eegs=brain_eegs,
                                         downsample=5,
                                         phase='val')

        ids.append(train.iloc[valid_index]['eeg_id'].values)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

        # get model
        model_path = f"{out_final_path}/best_model_fold{i}.pth"
        model = SpectroCNN(
            model_name=CFG.wave_model_name,
            pretrained=False,
            num_classes=6,
            spectrogram=WaveNetSpectrogram,
            spec_params=CFG.spec_params,
            resize_img=None,
            custom_classifier=CFG.custom_classifier,
            upsample=CFG.upsample,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))

        # inference
        val_pred = run_inference_loop(model, val_loader, device)
        # oof_pred_arr[valid_index] = val_pred
        all_oof.append(val_pred)

        model_path = f"{out_final_path}/best_model_fold{i}_ema.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        val_pred = run_inference_loop(model, val_loader, device)
        all_oof_ema.append(val_pred)

        all_true.append(train.iloc[valid_index][CLASSES].values)

        del valid_index
        del model, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_oof_ema = np.concatenate(all_oof_ema)
    all_true = np.concatenate(all_true)
    all_ids = np.concatenate(ids, axis=0)

    oof = pd.DataFrame(all_oof.copy())
    oof['eeg_id'] = all_ids

    oof_ema = pd.DataFrame(all_oof_ema.copy())
    oof_ema['eeg_id'] = all_ids

    true = pd.DataFrame(all_true.copy())
    true['eeg_id'] = all_ids

    true2 = pd.DataFrame(all_true.copy())
    true2['eeg_id'] = all_ids

    oof.to_csv(f'{out_final_path}/oof_{CFG.exp_num}.csv')
    oof_ema.to_csv(f'{out_final_path}/oof_ema_{CFG.exp_num}.csv')
    true.to_csv(f'{out_final_path}/gt_{CFG.exp_num}.csv')

    cv_score = score(solution=true, submission=oof, row_id_column_name='eeg_id')
    cv_score_ema = score(solution=true2, submission=oof_ema, row_id_column_name='eeg_id')
    print(f'CV Score KL-Div for {CFG.model_name}', cv_score)
    print(f'CV Score KL-Div for {CFG.model_name}, w ema', cv_score_ema)
    with open(f"{out_final_path}/res.txt", mode="w+") as f:
        f.write(f'CV Score KL-Div for {CFG.model_name}: {cv_score}\n')
        f.write(f'CV Score KL-Div for {CFG.model_name}: {cv_score_ema}\n')
    f.close()


if __name__ == "__main__":
    main()
