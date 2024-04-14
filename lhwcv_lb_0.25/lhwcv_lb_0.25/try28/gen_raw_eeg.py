# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import gc
import math
import os
from datetime import datetime
from pathlib import Path
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import numpy as np
import pandas as pd
import torch.nn.functional as F

DATA_PATH = Path("/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/")


class CFG:
    train_models = True
    seed = 42

    exp_id = 'try4_more_fea'  # datetime.now().strftime("%m%d-%H-%M-%S")
    exp_dump_path = Path("/home/lhw/m2_disk/kaggle/working/") / exp_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # == Data ==
    gen_eegs = True
    # Chris' 8 channels
    feats = [
        "Fp1", "T3", "C3", "O1", "F7", "T5", "F3", "P3",
        "Fp2", "C4", "T4", "O2", "F8", "T6", "F4", "P4",
        #"Fz", "Cz", "Pz"
    ]



N_CLASSES = 6
TGT_VOTE_COLS = [
    "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote",
    "grda_vote", "other_vote"
]
TGT_COL = "target"
EEG_FREQ = 200  # Hz
EEG_WLEN = 50  # sec
EEG_PTS = int(EEG_FREQ * EEG_WLEN)

def _get_eeg_window(file: Path) -> np.ndarray:
    """Return cropped EEG window.

    Default setting is to return the middle 50-sec window.

    Args:
        file: EEG file path
        test: if True, there's no need to truncate EEGs

    Returns:
        eeg_win: cropped EEG window
    """
    eeg = pd.read_parquet(file, columns=CFG.feats)
    n_pts = len(eeg)
    #offset = (n_pts - EEG_PTS) // 2
    #eeg = eeg.iloc[offset:offset + EEG_PTS]

    #eeg_win = np.zeros((EEG_PTS, len(CFG.feats)))
    eeg_win = np.zeros((n_pts, len(CFG.feats)))
    for j, col in enumerate(CFG.feats):
        if CFG.cast_eegs:
            eeg_raw = eeg[col].values.astype("float32")
        else:
            eeg_raw = eeg[col].values

            # Fill missing values
        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else:
            # All missing
            eeg_raw[:] = 0
        eeg_win[:, j] = eeg_raw

    return eeg_win


train = pd.read_csv(DATA_PATH / "train.csv")

uniq_eeg_ids = train["eeg_id"].unique()
n_uniq_eeg_ids = len(uniq_eeg_ids)

if True:
    all_eegs = {}
    for i, eeg_id in tqdm(enumerate(uniq_eeg_ids), total=n_uniq_eeg_ids):
        eeg_win = _get_eeg_window(DATA_PATH / "train_eegs" / f"{eeg_id}.parquet")
        all_eegs[eeg_id] = eeg_win
    np.save('./eegs_all.npy', all_eegs)
    exit(0)
