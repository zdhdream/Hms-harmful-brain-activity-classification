import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import lr_scheduler, AdamW
from torch.cuda import amp
import torchvision.models as models

import timm
from timm.utils import ModelEmaV2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from kaggle_kl_div import score

from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


class CFG:
    # base parameters
    exp_num = 3
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda:1"
    train_batch_size = 16
    val_batch_size = 32
    IMG_SIZE = [512, 512]
    # backbone && model parameters
    model_name = "tf_efficientnetv2_s.in1k"  # tf_efficientnet_b5
    max_len = 512
    in_channels = 3
    head_dropout = 0.2
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    # optimizer && scheduler parameters
    encoder_lr = 0.0005
    decoder_lr = 0.0005
    weight_decay = 0.001
    eps = 1.0e-06
    betas = [0.9, 0.999]
    warmupstep = 0
    # training parameters
    epochs = 20
    es_patience = 4
    # augmentation parameters
    mixup_out_prob = 0.5
    mixup_in_prob1 = 0.5
    mixup_in_prob2 = 0.5
    mixup_alpha_in = 5.0
    mixup_alpha_out = 5.0

    grad_acc = 1


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
        # A.Resize(p=1.0, height=CFG.height, width=CFG.width),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        # A.Resize(p=1.0, height=CFG.height, width=CFG.width),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform


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

    def smooth_labels(self, labels, factor=0.01):
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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


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
        self.fco = nn.Linear(in_features, num_classes)
        init_layer(self.fc1)
        init_layer(self.fco)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

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
        # eeg spectrograms
        # x2 = [x[:, i + 4:i + 5, :, :] for i in range(4)]
        # x2 = torch.concatenate(x2, dim=2)  # (bs, 1, 512, 256)
        # x = torch.concatenate([x1, x2], dim=3)  # (bs,1,512,512)
        return x1

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        x1 = self.operate(x1)
        x2 = self.operate(x2)
        x3 = self.operate(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)  # (bs,3,1536,512)
        x = self.spec_augmenter(x)
        x = self.model(x)

        xgem = self.gem_pooling(x)[:, :, 0, 0]

        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        xatt = (x * attn_weights).sum(dim=1)
        x = torch.concatenate([xgem, xatt], dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        x = self.fco(x)
        return x


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn: bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn: bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
                 freq_stripes_num):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width,
                                        stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width,
                                        stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class PANNsCNN14Att(nn.Module):

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            apply_aug: bool,
    ):
        super().__init__()
        self.apply_aug = apply_aug
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=512, out_channels=1024)

        self.bn0 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(1024, num_classes, activation='sigmoid')

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def cnn_feature_extractor(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        return x

    def operate(self, x):
        # kaggle spectrograms
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 1536)
        return x1

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        x1 = self.operate(x1)
        x2 = self.operate(x2)
        x3 = self.operate(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)  # (bs,3,1536,512)
        # x (bs, 3, time_steps, freq_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training and self.apply_aug:
            x = self.spec_augmenter(x)
        # x (bs, in_chans, time, frequency)
        x = self.cnn_feature_extractor(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x).transpose(1, 2))
        return logit


class PANNsDense121Att(nn.Module):

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            apply_aug: bool,
    ):
        super().__init__()
        self.apply_aug = apply_aug
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(1024, num_classes, activation='sigmoid')

        self.densenet_features = models.densenet121(pretrained=True).features

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def cnn_feature_extractor(self, x):
        x = self.densenet_features(x)
        return x

    def operate(self, x):
        # kaggle spectrograms
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 1536)
        return x1

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        x1 = self.operate(x1)
        x2 = self.operate(x2)
        x3 = self.operate(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)  # (bs,3,1536,512)
        # x (bs, 3, time_steps, freq_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training and self.apply_aug:
            x = self.spec_augmenter(x)
        # x (bs, in_chans, time, frequency)
        x = self.cnn_feature_extractor(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x).transpose(1, 2), dim=-1)
        return logit


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def get_optimizer(model, config):
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=config.encoder_lr,
                                                decoder_lr=config.decoder_lr,
                                                weight_decay=config.weight_decay
                                                )

    optimizer = AdamW(optimizer_parameters,
                      lr=config.encoder_lr,
                      eps=config.eps,
                      betas=config.betas)
    return optimizer


def train_one_fold(CFG, fold, train_index, val_index, train_all, spectrograms, all_eegs, output_path):
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(CFG)

    train_dataset = HMSHBASpecDataset(train_all.iloc[train_index],
                                      specs=spectrograms,
                                      eeg_specs=all_eegs,
                                      transform=train_transform, phase='train')

    val_dataset = HMSHBASpecDataset(train_all.iloc[val_index],
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

    optimizer = get_optimizer(model, CFG)

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
                with amp.autocast(use_amp, dtype=torch.float16):
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

            with torch.no_grad(), amp.autocast(use_amp, dtype=torch.float16):
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
    model.half()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data1"], device)
            x = x.half()
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

    train = train.reset_index()

    spectrograms = np.load(f"{DATA}/brain-spectrograms/specs.npy", allow_pickle=True).item()
    all_eegs = np.load(f"{DATA}/eeg_spectrograms/eeg_specs_256x512.npy", allow_pickle=True).item()

    score_list = []
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # gkf = GroupKFold(n_splits=N_FOLDS)
    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):
        output_path = Path(f"{CFG.exp_num:03d}_fold{i}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{i}]")
        score_list.append(
            train_one_fold(CFG, i, train_index, valid_index, train, spectrograms, all_eegs, output_path))
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


if __name__ == "__main__":
    main()
