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
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

import pywt
import librosa

from kaggle_kl_div import score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


################################################
################## Config #######################
################################################
class CFG:
    model_name = "resnet34d"
    height = 256
    width = 128
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 42
    deterministic = True
    enable_amp = True
    device = "cuda"

    USE_WAVELET = None
    DISPLAY = 4
    CREATE_SPECTROGRAMS = True


################################################
################## Model #######################
################################################
class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
    ):
        super().__init__()
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        print(pretrained_cfg)
        pretrained_cfg['file'] = r"/root/.cache/torch/hub/checkpoints/resnet34d_ra2-f8dcfcaf.pth"
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels,
            pretrained_cfg=pretrained_cfg
        )

    def forward(self, x):
        """x: (bs, 4*128, 256, 1)"""
        h = self.model(x)

        return h


################################################
################## Dataset #####################
################################################
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            data,
            eegs: tp.Dict,
            transform: A.Compose,
    ):
        self.data = data
        self.eegs = eegs
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        img = self.eegs[row.eeg_id]  # shape: (128, 256, 4)
        spec = [img[:, :, k] for k in range(4)]
        img = np.concatenate(spec, axis=0)  # (4*128,256)
        label = row[CLASSES].values.astype("float32")
        # # log transform
        # img = np.clip(img, np.exp(-4), np.exp(8))  # 在log前如果数据中存在非常小或者非常大的值，log转换可能导致数值溢出或浮点数表示不稳定
        # img = np.log(img)  # 加上Log用于减小数据的动态范围，对于音频数据，这种转换可以使得较大的振幅差异变得更加平滑

        # normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


################################################
##################  Loss  ######################
################################################
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


################################################
###########  Function for training  ############
################################################
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


def get_eeg_ids_label(val_fold, train_all: pd.DataFrame):
    """Get file path and target info."""

    train_idx = train_all[train_all["fold"] != val_fold].index.values
    val_idx = train_all[train_all["fold"] == val_fold].index.values
    labels = train_all[CLASSES].values
    eeg_ids = []
    for eeg_id in train_all['eeg_id'].values:
        eeg_ids.append(eeg_id)

    train_data = {
        "eeg_ids": [eeg_ids[idx] for idx in train_idx],
        "labels": [labels[idx].astype("float32") for idx in train_idx]}

    val_data = {
        "eeg_ids": [eeg_ids[idx] for idx in val_idx],
        "labels": [labels[idx].astype("float32") for idx in val_idx]}

    return train_data, val_data, train_idx, val_idx


def get_transforms(CFG):
    train_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.height, width=CFG.width),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.height, width=CFG.width),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform


################################################
##################  Training  ##################
################################################
def train_one_fold(CFG, val_fold, train_all, all_eegs, output_path):
    """Main"""
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(CFG)

    train_dataset = HMSHBACSpecDataset(train_all[train_all["fold"] != val_fold], eegs=all_eegs,
                                       transform=train_transform)
    val_dataset = HMSHBACSpecDataset(train_all[train_all["fold"] == val_fold], eegs=all_eegs,
                                     transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, num_workers=0, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.batch_size, num_workers=0, shuffle=False, drop_last=False)

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)
    model.to(device)

    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, epochs=CFG.max_epoch,
        pct_start=0.0, steps_per_epoch=len(train_loader),
        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    )

    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()

    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)

    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0

    for epoch in range(1, CFG.max_epoch + 1):
        epoch_start = time()
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]

            optimizer.zero_grad()
            with amp.autocast(use_amp):
                y = model(x)
                loss = loss_func(y, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
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

    return val_fold, best_epoch, best_val_loss


################################################
##############  Inference OOF  #################
################################################
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


################################################
##############  Main function  #################
################################################
def main():
    train = pd.read_csv(f"{DATA}/train.csv")
    train[CLASSES] /= train[CLASSES].sum(axis=1).values[:, None]

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(sgkf.split(train, y=train["expert_consensus"], groups=train["patient_id"])):
        train.loc[val_idx, "fold"] = fold_id

    all_eegs = np.load("data/eeg_spectrograms/eeg_specs.npy", allow_pickle=True).item()

    score_list = []
    for fold_id in FOLDS:
        output_path = Path(f"fold{fold_id}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{fold_id}]")
        score_list.append(train_one_fold(CFG, fold_id, train, all_eegs, output_path))
    print(score_list)

    # Save best model
    best_log_list = []
    for (fold_id, best_epoch, _) in score_list:

        exp_dir_path = Path(f"fold{fold_id}")
        best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
        copy_to = f"./best_model_fold{fold_id}.pth"
        shutil.copy(best_model_path, copy_to)

        for p in exp_dir_path.glob("*.pth"):
            p.unlink()

    # Inference Out of Fold
    label_arr = train[CLASSES].values
    oof_pred_arr = np.zeros((len(train), N_CLASSES))
    score_list = []

    for fold_id in range(N_FOLDS):
        print(f"\n[fold {fold_id}]")
        device = torch.device(CFG.device)

        # # get_dataloader
        _, val_path_label, _, val_idx = get_eeg_ids_label(fold_id, train)
        _, val_transform = get_transforms(CFG)
        val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=CFG.batch_size, num_workers=0, shuffle=False, drop_last=False)

        # # get model
        model_path = f"./best_model_fold{fold_id}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # # inference
        val_pred = run_inference_loop(model, val_loader, device)
        oof_pred_arr[val_idx] = val_pred

        del val_idx, val_path_label
        del model, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate OOF score
    true = train[["label_id"] + CLASSES].copy()

    oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
    oof.insert(0, "label_id", train["label_id"])

    cv_score = score(solution=true, submission=oof, row_id_column_name='label_id')
    print(f'CV Score KL-Div for {CFG.model_name}', cv_score)


if __name__ == "__main__":
    main()
