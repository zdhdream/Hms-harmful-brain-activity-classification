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
from torch.optim import AdamW
from torch.cuda import amp

import timm
from timm.models.layers import get_act_layer
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2

import albumentations as A
from albumentations.pytorch import ToTensorV2
# from transforms import *

from kaggle_kl_div import score

from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

from triplet_attention import TripletAttention

from scipy.signal import butter, lfilter, freqz
import scipy.signal as scisig

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
feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}


class CFG:
    # base parameters
    exp_num = 3
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda:0"
    train_batch_size = 16
    val_batch_size = 32
    IMG_SIZE = [512, 512]
    type = "wave"
    # backbone && model parameters
    model_name = "tf_efficientnetv2_s.in1k"  # tf_efficientnet_b5 tf_efficientnet_b2_ap tf_efficientnetv2_s.in1k
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
    lr = 8e-3
    lr_ratio = 5
    min_lr = 1e-6
    warmupstep = 0
    encoder_lr = 0.0005
    decoder_lr = 0.0005
    weight_decay = 0.001
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
    # wave model parameters
    wave_model_name = "f_efficientnetv2_s.in1k"
    base_filters = 128
    wave_layers = (10, 6, 2)
    kernel_size = 3
    resize_img = None
    custom_classifier = "gem"
    upsample = "bicubic"
    # wave model parameters v2
    fixed_kernel_size = 7
    linear_layer_features = 576
    kernels = [3, 5, 7, 11, 13]
    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов 10_000
    n_split_samples = 1
    out_samples = nsamples // n_split_samples  # 2_000
    sample_delta = nsamples - out_samples  # 8000
    sample_offset = sample_delta // 2

    bandpass_filter = {"low": 0.5, "high": 25, "order": 2}
    rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2
    random_divide_signal = 0.0
    random_close_zone = 0.0
    random_negative_signal = 0.0
    random_reverse_signal = 0.0
    random_common_negative_signal = 0.0
    random_common_reverse_signal = 0.0

    map_features = [
        ("Fp1", "T3"),
        ("T3", "O1"),
        ("Fp1", "C3"),
        ("C3", "O1"),
        ("Fp2", "C4"),
        ("C4", "O2"),
        ("Fp2", "T4"),
        ("T4", "O2"),
        # ('Fz', 'Cz'), ('Cz', 'Pz'),
    ]
    eeg_features = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]  # 'Fz', 'Cz', 'Pz'
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'
    n_map_features = len(map_features)
    w_in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)


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
        A.VerticalFlip(p=0.5),
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


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    return mu_x  # quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(
        data, cutoff_freq=20, sampling_rate=CFG.sampling_rate, order=4
):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def denoise_filter(x):
    # Частота дискретизации и желаемые частоты среза (в Гц).
    # Отфильтруйте шумный сигнал
    y = butter_bandpass_filter(x, CFG.lowcut, CFG.highcut, CFG.sampling_rate, order=6)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y


class HMSHBASpecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
            specs,
            eeg_specs,
            eegs,
            transform: A.Compose,
            downsample: int,
            bandpass_filter: tp.Dict[str, tp.Union[int, float]],
            rand_filter: tp.Dict[str, tp.Union[int, float]],
            phase: str,
    ):
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.eegs = eegs
        self.transform = transform
        self.downsample = downsample
        self.offset = None
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter
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

        X4 = self.__data_generation(index)
        if self.downsample is not None:
            X4 = X4[::self.downsample, :]

        y = (y1 + y2 + y3) / 3

        return {"data1": X1, "data2": X2, "data3": X3, "data4": X4, "target": y}

    def __data_generation(self, index: int):
        X = np.zeros((2_000, 8), dtype="float32")
        random_divide = False
        row = self.data.iloc[index]
        data = self.eegs[row.eeg_id]  # (10_000, 8)
        if self.cfg.nsamples != self.cfg.out_samples:
            if self.phase == "train":
                offset = (self.cfg.sample_delta * random.randint(0, 10_000)) // 1_000
            elif not self.offset is None:
                offset = self.offset
            else:
                offset = self.cfg.sample_offset

            if self.phase == "train" and self.cfg.random_divide_signal > 0.0 and random.uniform(0.0,
                                                                                                1.0) <= self.cfg.random_divide_signal:
                random_divide_signal = True
                multipliers = [(1, 2), (2, 3), (3, 4), (3, 5)]
                koef_1, koef_2 = multipliers[random.randint(0, 3)]
                offset = (koef_1 * offset) // koef_2
                data = data[offset:offset + (self.cfg.out_samples * koef_2) // koef_1, :]
            else:
                data = data[offset:offset + self.cfg.out_samples, :]

        reverse_signal = False
        negative_signal = False
        if self.phase == "train":
            if self.cfg.random_common_reverse_signal > 0.0 and random.uniform(0.0,
                                                                              1.0) <= self.cfg.random_common_reverse_signal:
                reverse_signal = True
            if self.cfg.random_common_negative_signal > 0.0 and random.uniform(0.0,
                                                                               1.0) <= self.cfg.random_common_negative_signal:
                negative_signal = True
            for i, (feat_a, feat_b) in enumerate(self.cfg.map_features):
                if self.phase == "train" and self.cfg.random_close_zone > 0.0 and random.uniform(0.0,
                                                                                                 1.0) <= self.cfg.random_close_zone:
                    continue
                diff_feat = (
                        data[:, self.cfg.feature_to_index[feat_a]] - data[:, self.cfg.feature_to_index[feat_b]]
                )  # (10_000,)

                if self.phase == "train":
                    if reverse_signal or self.cfg.random_reverse_signal > 0.0 and random.uniform(0.0,
                                                                                                 1.0) <= self.cfg.random_reverse_signal:
                        diff_feat = np.flip(diff_feat)
                    if negative_signal or self.cfg.random_negative_signal > 0.0 and random.uniform(0.0,
                                                                                                   1.0) <= self.cfg.random_negative_signal:
                        diff_feat = -diff_feat

                if not self.bandpass_filter is None:
                    diff_feat = butter_bandpass_filter(
                        diff_feat,
                        self.bandpass_filter["low"],
                        self.bandpass_filter["high"],
                        CFG.sampling_rate,
                        order=self.bandpass_filter["order"],
                    )

                if random_divide_signal:
                    # diff_feat = cp.asnumpy(cpsig.upfirdn([1.0, 1, 1.0], diff_feat, 2, 3))  # linear interp, rate 2/3
                    diff_feat = scisig.upfirdn([1.0, 1, 1.0], diff_feat, koef_1, koef_2)  # linear interp, rate 2/3
                    diff_feat = diff_feat[0:CFG.out_samples]

                if (
                        self.mode == "train"
                        and not self.rand_filter is None
                        and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
                ):
                    lowcut = random.randint(
                        self.rand_filter["low"], self.rand_filter["high"]
                    )
                    highcut = lowcut + self.rand_filter["band"]
                    diff_feat = butter_bandpass_filter(
                        diff_feat,
                        lowcut,
                        highcut,
                        self.cfg.sampling_rate,
                        order=self.rand_filter["order"],
                    )

                X[:, i] = diff_feat

        n = self.cfg.n_map_features
        if len(self.cfg.freq_channels) > 0:
            for i in range(self.cfg.n_map_features):
                diff_feat = X[:, i]
                for j, (lowcut, highcut) in enumerate(self.cfg.freq_channels):
                    band_feat = butter_bandpass_filter(
                        diff_feat, lowcut, highcut, self.cfg.sampling_rate, order=self.cfg.filter_order,  # 6
                    )
                    X[:, n] = band_feat
                    n += 1

        for spml_feat in self.cfg.simple_features:
            feat_val = data[:, self.cfg.feature_to_index[spml_feat]]

            if not self.bandpass_filter is None:
                feat_val = butter_bandpass_filter(
                    feat_val,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    self.cfg.sampling_rate,
                    order=self.bandpass_filter["order"],
                )

            if (
                    self.mode == "train"
                    and not self.rand_filter is None
                    and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                feat_val = butter_bandpass_filter(
                    feat_val,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, n] = feat_val
            n += 1

            # Обрезать края превышающие значения [-1024, 1024]
        X = np.clip(X, -1024, 1024)

        # Замените NaN нулем и разделить все на 32
        X = np.nan_to_num(X, nan=0) / 32.0

        if self.phase == 'train':
            # if torch.rand(1)<self.aug_prob:
            if torch.rand(1) < 0.5:
                X = np.flip(X, 0)  # sample[::-1]
            # if torch.rand(1)<self.aug_prob:
            if torch.rand(1) < 0.5:
                X = X * -1
        # обрезать полосовым фильтром верхнюю границу в 20 Hz.
        # X = butter_lowpass_filter(X, order=CFG.filter_order)  # 4
        return X

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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


# v1
class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            path=None,
    ):
        super().__init__()
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        print(pretrained_cfg)
        pretrained_cfg['file'] = r"/root/.cache/torch/hub/checkpoints/tf_efficientnetv2_s-eb54923e.pth"
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=CFG.backbone_dropout,
            drop_path_rate=CFG.backbone_droppath,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
            pretrained_cfg=pretrained_cfg
        )
        if path is not None:
            self.model.load_state_dict(torch.load(path))
        in_features = self.model.num_features
        self.fc1 = nn.Linear(2 * in_features, in_features)
        self.fco = nn.Linear(in_features, in_features)
        init_layer(self.fc1)
        init_layer(self.fco)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )

        self.dropout1 = nn.Dropout(p=CFG.head_dropout)
        self.dropout2 = nn.Dropout(p=CFG.head_dropout)

        self.lrelu = nn.LeakyReLU(0.1)

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def gem_pooling_1d(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)

    def avg_pooling(self, x, p=1, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def operate(self, x):
        # kaggle spectrograms
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 1536)
        return x1

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        x1 = self.operate(x1)  # (bs,1,1536, 512)
        x2 = self.operate(x2)
        x3 = self.operate(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)  # (bs,3,1536,512)

        x = self.model(x)

        xgem = self.gem_pooling(x)[:, :, 0, 0]
        # ygem = self.gem_pooling_1d(y)[:, :, 0]

        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        xatt = (x * attn_weights).sum(dim=1)
        # x = torch.concatenate([xgem, ygem, xatt], dim=1)
        x = torch.concatenate([xgem, xatt], dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        x = self.fco(x)
        return x


# EEGNet
class ResNet_1D_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            downsampling,
            dilation=1,
            groups=1,
            dropout=0.1,
            depth=0
    ):
        super(ResNet_1D_Block, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        ) if depth % 2 == 1 else nn.Identity()
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out = out + identity
        return out


class EEGNet(nn.Module):
    def __init__(
            self,
            kernels,
            in_channels,
            fixed_kernel_size,
            num_classes,
            linear_layer_features,
            dilation=1,
            groups=1,
    ):
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 32
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=kernel_size // 2,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes * len(self.kernels))
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes * len(self.kernels),
            out_channels=self.planes * len(self.kernels),
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes * len(self.kernels))
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            # dropout=0.2,
        )

        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            dilation=1,
            groups=1,
            blocks=19,
            padding=0,
            dropout=0.1,
    ):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            ) if i % 2 == 1 else nn.Identity()
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes * len(self.kernels),
                    out_channels=self.planes * len(self.kernels),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                    depth=i
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)

        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=1)
        # out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out.mean(1)  # [:, -1, :]

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        new_out = self.extract_features(x)
        # result = self.fc(new_out)
        return new_out


class HMSCustomModel(nn.Module):
    def __init__(self,
                 model_name,
                 pretrained,
                 s_in_channels,
                 num_classes,
                 kernels,
                 w_in_channels,
                 fixed_kernel_size,
                 linear_layer_features,
                 dilation=1,
                 group=1,
                 ):
        self.SpecModel = HMSHBACSpecModel(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_channels=s_in_channels,
        )
        self.WaveModel = EEGNet(
            kernels=kernels, in_channels=w_in_channels, fixed_kernel_size=fixed_kernel_size, num_classes=num_classes,
            linear_layer_features=linear_layer_features,
        )
        self.fc = nn.Linear(1,num_classes)

    def forward(self, x1, x2, x3, x4):
        x1 = self.SpecModel(x1,x2,x3)
        x2 = self.WaveModel(x4)
        x = torch.concatenate()



def get_optimizer(model, learning_rate, ratio, decay=1e-2):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': decay, "lr": learning_rate / ratio},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": learning_rate / ratio}
    ]
    return AdamW(optimizer_grouped_parameters)


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


def train_one_fold(CFG, fold, train_index, val_index, train_all, spectrograms, all_eegs, brain_eegs,output_path):
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(CFG)

    data, val = train_all.iloc[train_index], train_all.iloc[val_index]

    train_dataset = HMSHBASpecDataset(data,
                                      specs=spectrograms,
                                      eeg_specs=all_eegs,
                                      transform=train_transform, phase='train')

    val_dataset = HMSHBASpecDataset(val,
                                    specs=spectrograms,
                                    eeg_specs=all_eegs,
                                    transform=val_transform, phase='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_batch_size, num_workers=N_WORKERS, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels)

    model.to(device)

    model_ema = ModelEmaV2(model, decay=0.99)

    optimizer = get_optimizer(
        model,
        CFG.lr,
        CFG.lr_ratio
    )

    warmup_steps = CFG.epochs / 10 * len(train_loader) // CFG.grad_acc
    num_total_steps = CFG.epochs * len(train_loader) // CFG.grad_acc
    num_cycles = 0.48

    print(warmup_steps, num_total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_total_steps,
                                                num_cycles=num_cycles)

    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()

    loss_func_val_ema = KLDivLossWithLogitsForVal()

    use_amp = CFG.enable_amp
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
                x1, x2, x3, t = batch["data1"], batch["data2"], batch["data3"], batch["target"]

                optimizer.zero_grad()
                with amp.autocast(use_amp, dtype=torch.float32):
                    y = model(x1, x2, x3)
                    loss = loss_func(y, t)

                train_loss += loss.item()
                if CFG.grad_acc > 1:
                    loss = loss / CFG.grad_acc
                scaler.scale(loss).backward()

                if (idx + 1) % CFG.grad_acc == 0:
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
                        loss=f'{loss.item() * CFG.grad_acc:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )

                model_ema.update(model)

        train_loss /= len(train_loader)

        model.eval()
        for batch in tqdm(val_loader):
            x, t = batch["data1"], batch["target"]

            x = to_device(x, device)

            with torch.no_grad(), amp.autocast(use_amp, dtype=torch.float32):
                y = model(x, x, x)
                y_ema = model_ema.module(x, x, x)
            y = y.detach().cpu().to(torch.float32)
            y_ema = y_ema.detach().cpu().to(torch.float32)

            loss_func_val(y, t)
            loss_func_val_ema(y_ema, t)

        val_loss = loss_func_val.compute()
        val_loss_ema = loss_func_val_ema.compute()

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss

        if val_loss_ema < best_val_loss_ema:
            best_epoch_ema = epoch
            best_val_loss_ema = val_loss_ema

        elapsed_time = time() - epoch_start
        print(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, val loss ema: {val_loss_ema: .6f}, elapsed_time: {elapsed_time: .3f}sec")

        if epoch - best_epoch > CFG.es_patience and epoch - best_epoch_ema > CFG.es_patience:
            print("Early Stopping!")
            break

        train_loss = 0

    train_dataset2 = HMSHBASpecDataset(data[data['kl'] < 5.5],
                                       specs=spectrograms,
                                       eeg_specs=all_eegs,
                                       transform=train_transform, phase='train')

    val_dataset2 = HMSHBASpecDataset(val[val['kl'] < 5.5],
                                     specs=spectrograms,
                                     eeg_specs=all_eegs,
                                     transform=val_transform, phase='val')

    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=CFG.train_batch_size, num_workers=N_WORKERS, shuffle=True, drop_last=True)
    val_loader2 = torch.utils.data.DataLoader(
        val_dataset2, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

    best_val_loss = 1
    best_val_loss_ema = 1
    best_epoch = 0
    best_epoch_ema = 0
    train_loss = 0

    for epoch in range(1, CFG.epochs_2 + 1):
        epoch_start = time()
        model.train()
        with tqdm(train_loader2, leave=True) as pbar:
            for idx, batch in enumerate(pbar):
                batch = to_device(batch, device)
                x1, x2, x3, t = batch["data1"], batch["data2"], batch["data3"], batch["target"]

                optimizer.zero_grad()
                with amp.autocast(use_amp, dtype=torch.float32):
                    y = model(x1, x2, x3)
                    loss = loss_func(y, t)

                train_loss += loss.item()
                if CFG.grad_acc > 1:
                    loss = loss / CFG.grad_acc
                scaler.scale(loss).backward()

                if (idx + 1) % CFG.grad_acc == 0:
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
                        loss=f'{loss.item() * CFG.grad_acc:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )

                model_ema.update(model)

        train_loss /= len(train_loader)

        model.eval()
        for batch in tqdm(val_loader2):
            x, t = batch["data1"], batch["target"]

            x = to_device(x, device)

            with torch.no_grad(), amp.autocast(use_amp, dtype=torch.float32):
                y = model(x, x, x)
                y_ema = model_ema.module(x, x, x)
            y = y.detach().cpu().to(torch.float32)
            y_ema = y_ema.detach().cpu().to(torch.float32)

            loss_func_val(y, t)
            loss_func_val_ema(y_ema, t)

        val_loss = loss_func_val.compute()
        val_loss_ema = loss_func_val_ema.compute()

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            if CFG.device != 'cuda:0':
                model.to('cuda:0')
            torch.save(model.state_dict(), str(f"{output_path}/snapshot_epoch_{epoch}.pth"))
            if CFG.device != 'cuda:0':
                model.to(CFG.device)

        if val_loss_ema < best_val_loss_ema:
            best_epoch_ema = epoch
            best_val_loss_ema = val_loss_ema
            if CFG.device != 'cuda:0':
                model_ema.to('cuda:0')
            torch.save(model_ema.module.state_dict(), str(f"{output_path}/snapshot_epoch_{epoch}_ema.pth"))
            if CFG.device != 'cuda:0':
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
            x = to_device(batch["data1"], device)
            # x = x.half()
            y = model(x, x, x)
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

    spectrograms = np.load(f"{DATA}/brain-spectrograms/specs.npy", allow_pickle=True).item()
    all_eegs = np.load(f"{DATA}/eeg_spectrograms/eeg_scalos_256x512_mix.npy", allow_pickle=True).item()
    brain_eegs = np.load(f"{DATA}/brain-eegs/eegs.npy", allow_pickle=True).item()

    score_list = []
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # gkf = GroupKFold(n_splits=N_FOLDS)
    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):
        output_path = Path(f"{CFG.exp_num:03d}_fold{i}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{i}]")
        score_list.append(
            train_one_fold(CFG, i, train_index, valid_index, train, spectrograms, all_eegs, brain_eegs,output_path))
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
        val_dataset = HMSHBASpecDataset(
            train.iloc[valid_index],
            specs=spectrograms,
            eeg_specs=all_eegs,
            transform=val_transform, phase='val'
        )

        ids.append(train.iloc[valid_index]['eeg_id'].values)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

        # get model
        model_path = f"{out_final_path}/best_model_fold{i}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
        # model = HMSHBACSpecModelSED(
        #     model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels
        # )
        # model = BlockAttentionModel(
        #     model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels
        # )
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
