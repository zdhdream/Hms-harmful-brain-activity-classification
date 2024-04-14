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

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from madgrad import MADGRAD

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

import pywt
import librosa

from kaggle_kl_div import score

from tqdm import tqdm

from transformers import AutoModel, AutoConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification

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

# N_WORKERS = os.cpu_count()
N_WORKERS = 0


class CFG:
    # base parameters
    exp_num = 14
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda:0"
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
    lr = 4e-3 / 2
    lr_ratio = 5
    min_lr = 4e-5 / 2
    warmupstep = 0
    # training parameters
    epochs = 15
    es_patience = 4
    # augmentation parameters
    mixup_out_prob = 0.5
    mixup_in_prob1 = 0.5
    mixup_in_prob2 = 0.5
    mixup_alpha_in = 5.0
    mixup_alpha_out = 5.0


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

    def __len__(self):
        return len(self.data)

    def make_Xy(self, new_ind):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((512, 512), dtype="float32")

        row = self.data.iloc[new_ind]
        r = int((row['min'] + row['max']) // 4)

        # for k in range(1):
        # extract transform spectrogram
        # img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T  # (100, 300)
        img = self.specs[row.spec_id][:, :].T  # (256, 512 or ???)  [w, h] -> [h, w]
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

        # The 0th channel stores kaggle spectrogram
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (256, 512, 4)
        # 1th~3th channel store eeg spectrogram
        X[0:256, :, 1] = img[:, :, 0]  # (512, 512, 3)
        X[256:512, :, 1] = img[:, :, 1]
        X[0:256, :, 2] = img[:, :, 2]
        X[256:512, :, 2] = img[:, :, 3]

        X = self.spec_mask(X)

        if torch.rand(1) > self.aug_prob:
            X = self.shift_img(X)

        X = self._apply_transform(X)

        y[:] = row[CLASSES]

        return X, y

    def __getitem__(self, index: int):

        X1, y1 = self.make_Xy(index)  # X1 (512,512,3)
        # a pseudo-MixUp
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
        """tTime axis shift """
        s = torch.randint(0, CFG.IMG_SIZE[1], (1,))[0]
        new = np.concatenate([img[:, s:], img[:, :s]], axis=1)
        return new

    def spec_mask(self, img, max_it=4):
        count = 0
        new = img
        # frequency mask
        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[0] - CFG.IMG_SIZE[0] // 16, (1,))[0]
            h = torch.randint(CFG.IMG_SIZE[0] // 32, CFG.IMG_SIZE[0] // 16, (1,))[0]
            new[s:s + h] *= 0
            count += 1

        count = 0
        # time mask
        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[1] - CFG.IMG_SIZE[1] // 16, (1,))[0]
            w = torch.randint(CFG.IMG_SIZE[1] // 32, CFG.IMG_SIZE[1] // 16, (1,))[0]
            new[:, s:s + w] *= 0
            count += 1
        return new


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def compute_deltas(
        specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate"
) -> torch.Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )

    output = (
            torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


def make_delta(input_tensor: torch.Tensor):
    """
    计算时间维度的增量特征
    :param input_tensor: (bs, channels, n_mels, n_frames)
    :return:
    """
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


def image_delta(x):
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    x = torch.cat([x, delta_1, delta_2], dim=1)
    return x


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super(AttBlockV2, self).__init__()
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
        # x: (bs, 512, 512)
        # attention weights
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # nonlinear transform
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class HMSHBACSpecModelSED(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            stage: str,
    ):
        super().__init__()
        self.cfg = CFG
        self.in_chans = in_channels
        self.num_classes = num_classes
        self.stage = stage
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        print(pretrained_cfg)
        pretrained_cfg['file'] = r"/root/.cache/torch/hub/checkpoints/tf_efficientnetv2_s-eb54923e.pth"
        base_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=CFG.backbone_dropout,
            drop_path_rate=CFG.backbone_droppath,
            in_chans=in_channels,
            pretrained_cfg=pretrained_cfg
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if "efficientnet" in self.cfg.model_name:
            in_features = base_model.classifier.in_features
        elif "eca" in self.cfg.model_name:
            in_features = base_model.head.fc.in_features
        elif "res" in self.cfg.model_name:
            in_features = base_model.fc.in_features
        self.bn0 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def freeze(self):
        if self.stage == "finetune":
            self.encoder.eval()
            self.fc1.eval()
            self.bn0.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.bn0.parameters():
                param.requires_grad = False
        return

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def avg_pooling(self, x, p=1, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def operate(self, x):
        # kaggle spectrograms
        # print(x.shape)
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 1536)
        # eeg spectrograms
        # x2 = [x[:, i + 4:i + 5, :, :] for i in range(4)]
        # x2 = torch.concatenate(x2, dim=2)  # (bs, 1, 512, 256)
        # x = torch.concatenate([x1, x2], dim=3)  # (bs,1,512,512)
        return x1

    def extract_features(self, x):
        """
        :param x: (bs, n_channles, n_mels, n_frames)
        :return:
        """
        x = x.permute((0, 1, 3, 2))  # (bs, n_channles, n_frames, n_mels)
        frames_num = x.shape[2]

        x = x.transpose(1, 3)  # (bs, n_mels, n_frames, n_channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (bs, n_channels, n_frames, n_mels)

        x = x.transpose(2, 3)  # (bs, channles, n_mels, n_frames)
        x = self.encoder(x)

        # (bs, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num

    def forward(self, x1, x2, x3):
        x1 = self.operate(x1)
        x2 = self.operate(x2)
        x3 = self.operate(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)  # (bs,3,512,512)
        if self.in_chans == 3:
            x = image_delta(x)
        x, frames_num = self.extract_features(x)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x).transpose(1, 2))
        return logit


def get_optimizer(model, learning_rate, ratio, decay=1e-5):
    return MADGRAD(params=[
        {"params": model.encoder.parameters(), "lr": learning_rate / ratio},
        {"params": model.fc1.parameters(), "lr": learning_rate},
        {"params": model.att_block.parameters(), "lr": learning_rate}

    ], weight_decay=decay)


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

    # model = HMSHBACSpecModel(
    #     model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels)

    model = HMSHBACSpecModelSED(
        model_name=CFG.model_name,
        pretrained=True,
        num_classes=6,
        in_channels=CFG.in_channels,
        stage="finetune"
    )

    model.to(device)

    model_ema = ModelEmaV2(model, decay=0.99)

    optimizer = get_optimizer(
        model,
        CFG.lr,
        CFG.lr_ratio
    )

    scheduler = get_scheduler(
        optimizer,
        epochs=CFG.epochs,
        min_lr=CFG.min_lr,
        warmupstep=CFG.warmupstep
    )

    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()

    loss_func_val_ema = KLDivLossWithLogitsForVal()

    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)

    best_val_loss = 0.7  # 1.0e+09
    best_val_loss_ema = 0.65  # 1e9
    best_epoch = 0
    best_epoch_ema = 0
    train_loss = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time()
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            batch = to_device(batch, device)
            x1, x2, x3, t = batch["data1"], batch["data2"], batch["data3"], batch["target"]

            optimizer.zero_grad()
            with amp.autocast(use_amp, dtype=torch.float16):
                y = model(x1, x2, x3)
                loss = loss_func(y, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step(epoch + idx / len(train_loader))
            train_loss += loss.item()

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
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data1"], device)
            y = model(x, x, x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

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

    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    oof_ema = pd.DataFrame(all_oof_ema.copy())
    oof_ema['id'] = np.arange(len(oof_ema))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    true2 = pd.DataFrame(all_true.copy())
    true2['id'] = np.arange(len(true2))
    # Calculate OOF score
    # true = train[["label_id"] + CLASSES].copy()
    #
    # oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
    # oof.insert(0, "label_id", train["label_id"])

    oof.to_csv(f'{out_final_path}/oof_{CFG.exp_num}.csv')
    oof_ema.to_csv(f'{out_final_path}/oof_ema_{CFG.exp_num}.csv')
    true.to_csv(f'{out_final_path}/gt_{CFG.exp_num}.csv')

    cv_score = score(solution=true, submission=oof, row_id_column_name='id')
    cv_score_ema = score(solution=true2, submission=oof_ema, row_id_column_name='id')
    print(f'CV Score KL-Div for {CFG.model_name}', cv_score)
    print(f'CV Score KL-Div for {CFG.model_name}, w ema', cv_score_ema)


if __name__ == "__main__":
    main()
