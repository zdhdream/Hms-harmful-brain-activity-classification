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
from madgrad import MADGRAD

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

import pywt
import librosa

from kaggle_kl_div import score

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

RANDOM_SEED = 42
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)


class CFG:
    # base parameters
    seed = 42
    deterministic = True
    enable_amp = True
    device = "cuda"
    train_batch_size = 32
    val_batch_size = 32
    # backbone && model parameters
    model_name = "tf_efficientnet_b0"  # tf_efficientnet_b5
    in_channels = 3
    head_dropout = 0.2
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    # optimizer && scheduler parameters
    lr = 5e-3
    lr_ratio = 5
    min_lr = 5e-5
    warmupstep = 0
    # training parameters
    epochs = 9
    es_patience = 5
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
    ):
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((128, 256), dtype="float32")

        row = self.data.iloc[index]
        r = int((row['min'] + row['max']) // 4)

        for k in range(4):
            # extract transform spectrogram
            img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T  # (100, 300)

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
            X[14:-14, :, k] = img[:, 22:-22] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (128, 256, 4)
        X[:, :, 4:] = img  # (128, 256, 8)

        X = self._apply_transform(X)

        y[:] = row[CLASSES]

        return {"data": X, "target": y}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


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
        pretrained_cfg['file'] = r"/root/.cache/torch/hub/checkpoints/tf_efficientnet_b0_ns-c0e6a31c.pth"
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
        self.fc = nn.Linear(2 * in_features, num_classes)
        init_layer(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.dropout = nn.Dropout(p=CFG.head_dropout)

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        batch_size = x.shape[0]
        # kaggle spectrograms
        x1 = [x[:, i:i + 1, :, :] for i in range(4)]  # x: [bs,8,128,256]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 256)
        # eeg spectrograms
        x2 = [x[:, i + 4:i + 5, :, :] for i in range(4)]
        x2 = torch.concatenate(x2, dim=2)  # (bs, 1, 512, 256)
        x = torch.concatenate([x1, x2], dim=3)  # (bs,1,512,512)
        x = torch.concatenate([x, x, x], dim=1)  # (bs,3,512,512)
        x = self.model(x)
        xgem = self.gem_pooling(x)[:, :, 0, 0]
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        xatt = (x * attn_weights).sum(dim=1)
        x = torch.concatenate([xgem, xatt], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_optimizer(model, learning_rate, ratio, decay=0):
    return MADGRAD(params=[
        {"params": model.model.parameters(), "lr": learning_rate / ratio},
        {"params": model.fc.parameters(), "lr": learning_rate},
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
                                      transform=train_transform)

    val_dataset = HMSHBASpecDataset(train_all.iloc[val_index],
                                    specs=spectrograms,
                                    eeg_specs=all_eegs,
                                    transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_batch_size, num_workers=0, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.val_batch_size, num_workers=0, shuffle=False, drop_last=False)

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels)
    model.to(device)

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

    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)

    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time()
        model.train()
        for idx, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]

            optimizer.zero_grad()
            with amp.autocast(use_amp):
                y = model(x)
                loss = loss_func(y, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step(epoch + idx / len(train_loader))
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        for batch in val_loader:
            x, t = batch["data"], batch["target"]
            x = to_device(x, device)
            with torch.no_grad(), amp.autocast(use_amp):
                y = model(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            # print("save model")
            torch.save(model.state_dict(), str(f"{output_path}/snapshot_epoch_{epoch}.pth"))

        elapsed_time = time() - epoch_start
        print(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}")

        if epoch - best_epoch > CFG.es_patience:
            print("Early Stopping!")
            break

        train_loss = 0

    return fold, best_epoch, best_val_loss


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
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
    all_eegs = np.load(f"{DATA}/eeg_spectrograms/eeg_specs.npy", allow_pickle=True).item()

    score_list = []
    # sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
        output_path = Path(f"fold{i}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{i}]")
        score_list.append(
            train_one_fold(CFG, i, train_index, valid_index, train, spectrograms, all_eegs, output_path))
    print(score_list)

    for (fold_id, best_epoch, _) in score_list:
        exp_dir_path = Path(f"fold{fold_id}")
        best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
        copy_to = f"./best_model_fold{fold_id}.pth"
        shutil.copy(best_model_path, copy_to)

        for p in exp_dir_path.glob("*.pth"):
            p.unlink()

    # Inference Out of Fold

    all_oof = []
    all_true = []

    for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
        print(f"\n[fold {i}]")
        device = torch.device(CFG.device)

        # get transform
        _, val_transform = get_transforms(CFG)

        # get_dataloader
        val_dataset = HMSHBASpecDataset(
            train.iloc[valid_index],
            specs=spectrograms,
            eeg_specs=all_eegs,
            transform=val_transform
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=CFG.val_batch_size, num_workers=0, shuffle=False, drop_last=False)

        # get model
        model_path = f"./best_model_fold{i}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # inference
        val_pred = run_inference_loop(model, val_loader, device)
        # oof_pred_arr[valid_index] = val_pred
        all_oof.append(val_pred)
        all_true.append(train.iloc[valid_index][CLASSES].values)

        del valid_index
        del model, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))
    # Calculate OOF score
    # true = train[["label_id"] + CLASSES].copy()
    #
    # oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
    # oof.insert(0, "label_id", train["label_id"])
    cv_score = score(solution=true, submission=oof, row_id_column_name='id')
    print(f'CV Score KL-Div for {CFG.model_name}', cv_score)


if __name__ == "__main__":
    main()
