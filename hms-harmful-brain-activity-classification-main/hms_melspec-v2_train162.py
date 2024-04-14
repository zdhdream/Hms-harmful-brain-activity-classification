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

from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA = "hms-harmful-brain-activity-classification"
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

N_WORKERS = os.cpu_count() // 2


class CFG:
    # base parameters
    exp_num = 162
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda:0"
    train_batch_size = 16
    val_batch_size = 32
    IMG_SIZE = [512, 512]
    # backbone && model parameters
    model_name = "tf_efficientnet_b3.ns_jft_in1k"
    max_len = 512
    in_channels = 3
    head_dropout = 0.2
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    freeze = False
    frozen_layer_pct = 0.
    # transformer settings
    transformer_name = 'cross-encoder/ms-marco-MiniLM-L-2-v2'
    # optimizer && scheduler parameters
    lr = 4e-3 / 2
    lr_ratio = 5
    min_lr = 4e-5 / 2
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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        lams = []
        inv_lams = []
        for _ in range(batch_size):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            lams.append(lam)
            inv_lams.append(1.0 - lam)
        return torch.tensor(lams, dtype=torch.float32), torch.tensor(inv_lams, dtype=torch.float32)


class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            path=None,
            transformer_name=None
    ):
        super().__init__()
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        print(pretrained_cfg)
        # pretrained_cfg['file'] = r"/root/.cache/torch/hub/checkpoints/tf_efficientnet_b0_ns-c0e6a31c.pth"
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

        if CFG.freeze:
            n_freeze = int(len(list(self.model.named_parameters())) * CFG.frozen_layer_pct)
            print(
                f'Model {model_name} has {len(list(self.model.named_parameters()))} layers and going to freeze {n_freeze} layers')
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:n_freeze]):
                param.requires_grad = False

        in_features = self.model.num_features

        self.config = AutoConfig.from_pretrained(transformer_name, add_pooling_layer=False)
        self.transformer_encoder = AutoModel.from_pretrained(transformer_name)

        self.transformer_encoder = self.transformer_encoder.encoder
        self.inter_conv = nn.Conv1d(16 * 3 * in_features, self.config.hidden_size, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(2 * in_features + self.config.hidden_size, in_features)
        self.fco = nn.Linear(in_features, 6)
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

        # self.rnn = nn.LSTM(384, hidden_size=384//2, num_layers=2, batch_first=True, bidirectional=True)

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
        x = self.model(x)
        t = x.reshape(batch_size, x.size(1) * x.size(2), x.size(-1))
        xgem = self.gem_pooling(x)[:, :, 0, 0]

        t = self.inter_conv(t).permute(0, 2, 1)

        # rout = self.rnn(t)
        # rgem = self.gem_pooling_1d(rout)[:, :, 0]

        att_m = None  # torch.ones((t.size(0), 16)).to(CFG.device)
        tout = self.transformer_encoder(t, att_m)
        tout = tout.last_hidden_state.permute(0, 2, 1)
        tgem = self.gem_pooling_1d(tout)[:, :, 0]

        # print(xgem.shape)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       x = x.permute(0, 2, 1)
        # print(x.shape)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        xatt = (x * attn_weights).sum(dim=1)
        # x = torch.concatenate([xgem, ygem, xatt], dim=1)
        # print(xgem.shape, xatt.shape, tgem.shape)
        x = torch.concatenate([xgem, xatt, tgem], dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        x = self.fco(x)
        return x


def get_optimizer_mg(model, learning_rate, ratio, decay=0):
    return MADGRAD(params=[
        {"params": model.model.parameters(), "lr": learning_rate / ratio},
        {"params": model.fc1.parameters(), "lr": learning_rate},
        {"params": model.fco.parameters(), "lr": learning_rate},
        {"params": model.attention.parameters(), "lr": learning_rate},
        # {"params": model.wave_block1.parameters(), "lr": learning_rate},
        # {"params": model.wave_block2.parameters(), "lr": learning_rate},
        # {"params": model.wave_block3.parameters(), "lr": learning_rate},
        # {"params": model.wave_block4.parameters(), "lr": learning_rate},
        # {"params": model.LSTM.parameters(), "lr": learning_rate},
    ], weight_decay=decay)


from transformers import AdamW


# from torch.optim import AdamW

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
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels,
        transformer_name=CFG.transformer_name)

    model.to(device)

    model_ema = ModelEmaV2(model, decay=0.99)

    model_name_sep = CFG.transformer_name.split('/')[-1]
    torch.save(model.config, f'./{model_name_sep}_config.pth')

    optimizer = get_optimizer(
        model,
        CFG.lr,
        CFG.lr_ratio
    )

    # scheduler = get_scheduler(
    #     optimizer,
    #     epochs=CFG.epochs,
    #     min_lr=CFG.min_lr,
    #     warmupstep=CFG.warmupstep
    # )

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
                with amp.autocast(use_amp, dtype=torch.bfloat16):
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

            with torch.no_grad(), amp.autocast(use_amp, dtype=torch.bfloat16):
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

    train = train.reset_index()

    spectrograms = np.load(f"{DATA}/brain-spectrograms/specs.npy", allow_pickle=True).item()
    all_eegs = np.load(f"{DATA}/eeg_spectrograms/eeg_scalos_256x512.npy", allow_pickle=True).item()

    score_list = []
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # gkf = GroupKFold(n_splits=N_FOLDS)
    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):
        # if i!=0: exit(1)
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
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels,
            transformer_name=CFG.transformer_name)
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
    # Calculate OOF score
    # true = train[["label_id"] + CLASSES].copy()
    #
    # oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
    # oof.insert(0, "label_id", train["label_id"])

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