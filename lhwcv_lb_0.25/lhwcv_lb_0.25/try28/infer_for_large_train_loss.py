# -*- coding: utf-8 -*-
import tqdm
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

from dataset_v2 import DataGenerator
import numpy as np
from torch.utils.data import DataLoader

from raw_model3 import ECAPA_TDNN
import matplotlib.pyplot as plt

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
META = ['eeg_id', 'spectrogram_id', 'spectrogram_label_offset_seconds',
        'patient_id', 'expert_consensus', 'tag', 'total_votes',
        'spectrogram_label_offset_seconds',
        'eeg_label_offset_seconds',
        ]


def preprocess_df(df):
    print('origin df len: ', len(df))
    df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
    df2 = df2[df2['eeg_missing_ratio'] < 0.2]
    df = df2.copy().reset_index(drop=True)

    print('selected len by missing ratio: ', len(df))

    df['total_votes'] = df[TARGETS].values.sum(axis=1, keepdims=True)
    df[TARGETS] = df[TARGETS] / df[TARGETS].values.sum(axis=1, keepdims=True)
    return df


class KLDivWithLogitsLoss(nn.KLDivLoss):
    """Kullback-Leibler divergence loss with logits as input."""

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=1)
        kldiv_loss = super().forward(y_pred, y_true)

        return kldiv_loss


def predict_fn(model, val_dataloader: DataLoader,
               data_type='raw',
               return_loss=False,
               secs=50):
    model.eval()
    preds = []
    if return_loss:
        eval_loss = 0.0
    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad(), torch.cuda.amp.autocast():

            if data_type == 'hybrid':
                x1, x2, y = batch
                x1 = x1.cuda()
                x2 = x2.cuda()
                pred = model(x1, x2)
            else:
                x, y = batch
                if secs != 50:
                    lens = secs * 200
                    x = x[:, 5000 - lens // 2: 5000 + lens // 2]
                x = x.cuda()
                pred = model(x)

            # pred = torch.softmax(pred, dim=1)
            preds.append(pred.float().cpu().numpy())
    if return_loss:
        eval_loss = eval_loss / len(val_dataloader)
        return np.concatenate(preds), eval_loss
    return np.concatenate(preds)


if __name__ == '__main__':
    train = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train_tag.csv')
    train = preprocess_df(train)
    train = train.reset_index(drop=True)
    train = train[META + TARGETS]
    train.columns = ['eeg_id', 'spec_id', 'offset',
                     'patient_id', 'target', 'tag', 'total_votes',
                     'spectrogram_label_offset_seconds',
                     'eeg_label_offset_seconds',
                     ] + TARGETS
    #train = train[:320].copy()

    all_raw_eegs = np.load('../try4/eegs_all.npy',
                           allow_pickle=True).item()
    d = DataGenerator(train, mode='valid', specs=None,
                      eeg_specs=None, raw_eegs=all_raw_eegs,
                      data_type="raw")
    dloader = DataLoader(d, 128, False, num_workers=4)

    is_parallel = True if torch.cuda.device_count() > 1 else False
    if is_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    pred_list = []
    for i in range(5):
        model = ECAPA_TDNN('b2').cuda()
        model.load_state_dict(torch.load(f'v28_0329_all/raw/raw_spec_b2_50_10_5/raw_fold_{i + 1}_stage2_best.pth'))
        model = model.eval()
        if is_parallel:
            model = nn.DataParallel(model)

        pred = predict_fn(model, dloader)
        pred_list.append(pred)
    pred = np.mean(pred_list, axis=0)
    pred = torch.from_numpy(pred).float()
    targets = np.array(train[TARGETS].values)
    targets = torch.from_numpy(targets).float()

    losses = KLDivWithLogitsLoss().forward(pred, targets)
    losses = losses.mean(dim=-1).numpy()
    train = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train_tag.csv')
    train['loss'] = losses
    DATA_ROOT = '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/'
    train.to_csv(f'{DATA_ROOT}/train_tag_with_loss.csv')

    train['loss'].plot.hist(bins=100, title='loss')
    plt.savefig('loss.png')

