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

from torch.cuda import amp

from timm.scheduler import CosineLRScheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from kaggle_kl_div import score
from timm.utils import ModelEmaV3
from tqdm import tqdm
from cfg import CFG
from models import HMSHBACSpecModelSED, HMSHBACSpecModel
from datasets import HMSHBASpecDataset, HMSHBASpecDatasetADD, HMSHBASpecDatasetWithoutAug
from mix_up import Mixup

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA = "./"
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
METAv1 = ['spectrogram_id', 'spectrogram_label_offset_seconds', 'patient_id', 'expert_consensus']
METAv2 = ['eeg_id', 'spectrogram_id', 'spectrogram_label_offset_seconds',
          'patient_id', 'expert_consensus', 'tag', 'total_votes',
          'spectrogram_label_offset_seconds',
          'eeg_label_offset_seconds',
          'group_id'
          ]

N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)

N_WORKERS = 1  # os.cpu_count()

if not os.path.exists(CFG.exp_output_path):
    os.mkdir(CFG.exp_output_path)


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


class WeightedKLDivWithLogitsLoss(nn.KLDivLoss):
    def __init__(self, weight):
        super(WeightedKLDivWithLogitsLoss, self).__init__(size_average=None, reduce=None, reduction='none')
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        # TODO: For KLDivLoss: input should 'log-probability' and target should be 'probability'
        # TODO: input for this method is logits, and target is probabilities
        batch_size = input.size(0)
        log_prob = F.log_softmax(input, 1)
        element_loss = super(WeightedKLDivWithLogitsLoss, self).forward(log_prob, target)

        sample_loss = torch.sum(element_loss, dim=1)
        sample_weight = torch.sum(target * self.weight, dim=1)

        weighted_loss = sample_loss * sample_weight
        # Average over mini-batch, not element-wise
        avg_loss = torch.sum(weighted_loss) / batch_size

        return avg_loss


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


def get_optimizer(model, learning_rate, ratio, decay=0):
    return optim.AdamW(params=[
        {"params": model.model.parameters(), "lr": learning_rate / ratio},
        {"params": model.fc.parameters(), "lr": learning_rate},
    ], weight_decay=decay)  # optim.AdamW MADGRAD


# def get_optimizer(model, learning_rate, ratio, decay=0):
#     return optim.AdamW(params=[
#         {"params": model.encoder.parameters(), "lr": learning_rate / ratio},
#         {"params": model.fc1.parameters(), "lr": learning_rate},
#         {"params": model.att_block.parameters(), "lr": learning_rate},
#     ], weight_decay=decay)

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


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def train_one_fold(CFG, fold, train_index, val_index, train_all, spectrograms, all_eegs, aux_infos_by_group_id,
                   aux_infos_by_eeg_id, output_path):
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(CFG)

    train = train_all.iloc[train_index].reset_index(drop=True)
    val = train_all.iloc[val_index].reset_index(drop=True)

    # 意见相同1个段+意见相同多个段
    data1 = train[train['tag'].isin(['case1', 'case3'])].copy().reset_index(drop=True)
    # 意见不同1个段+意见不同多个段
    data2 = train[train['tag'].isin(['case2', 'case4'])].copy().reset_index(drop=True)
    # 从意见相同的所有数据中将total_votes＜5的全部过滤掉
    data3 = data1[data1['total_votes'] > CFG.t2].copy().reset_index(drop=True)
    # 从意见不同的所有数据中将total_votes<10的全部过滤掉
    data2 = data2[data2['total_votes'] > CFG.t1].copy().reset_index(drop=True)
    # 将过滤掉的数据重新拼接
    data2 = pd.concat([data2, data3]).reset_index(drop=True)

    val1 = val[val['tag'].isin(['case1', 'case3'])].copy().reset_index()
    val2 = val[val['tag'].isin(['case2', 'case4'])].copy().reset_index()
    val3 = val1[val1['total_votes'] > CFG.t2].copy().reset_index()
    val2 = val2[val2['total_votes'] > CFG.t1].copy().reset_index()
    val2 = pd.concat([val2, val3]).reset_index(drop=True)

    print('train len: {}'.format(len(train)))
    print('stage1 train ;en: {}'.format(len(train)))
    print('stage2 train len: {}'.format(len(data2)))

    print('stage1 val len: {}'.format(len(val)))
    print('stage2 val len: {}'.format(len(val2)))

    train_dataset = HMSHBASpecDataset(train,
                                      specs=spectrograms,
                                      eeg_specs=all_eegs,
                                      aux_infos_by_group_id=aux_infos_by_group_id,
                                      aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                      transform=train_transform, phase='train')

    val_dataset = HMSHBASpecDataset(val,
                                    specs=spectrograms,
                                    eeg_specs=all_eegs,
                                    aux_infos_by_group_id=aux_infos_by_group_id,
                                    aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                    transform=val_transform, phase='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_batch_size, num_workers=N_WORKERS, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels)
    current_model_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    if CFG.if_load_pretrained:
        loaded_state_dict = torch.load(f'./{CFG.pretrain_model_path}')
        model.load_state_dict(loaded_state_dict, strict=True)
    model.to(device)
    model_ema = ModelEmaV3(model, decay=0.99, device=None, use_warmup=True, warmup_power=3 / 4)
    optimizer = get_optimizer(
        model,
        CFG.lr,
        CFG.lr_ratio
    )
    mixup_args = {
        'mixup_alpha': CFG.mixup_alpha,
        'cutmix_alpha': CFG.cutmix_alpha,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': CFG.switch_prob,
        'mode': CFG.mode,
        'label_smoothing': 0,
        'num_classes': 6}
    mixup_fn = Mixup(**mixup_args)
    scheduler = get_scheduler(
        optimizer,
        epochs=CFG.epochs,
        min_lr=CFG.min_lr,
        warmupstep=CFG.warmupstep
    )
    # scheduler = lr_scheduler.OneCycleLR(
    #     optimizer=optimizer, epochs=CFG.epochs,
    #     pct_start=0.0, steps_per_epoch=len(train_loader),
    #     max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    # )
    class_weight_torch = torch.from_numpy(CFG.loss_weight).float()
    loss_func = KLDivLossWithLogits()  # WeightedKLDivWithLogitsLoss(class_weight_torch)
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
        for idx, batch in enumerate(tqdm(train_loader)):
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]
            # x, t = mixup_fn(x,t)
            optimizer.zero_grad()
            with amp.autocast(use_amp, dtype=torch.bfloat16):
                y_0 = model(x)
                loss_0 = loss_func(y_0, t)

                loss = loss_0  # 0.5 * (loss_0 + loss_1) + alpha * loss_r

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema.update(model)
            if scheduler is not None:
                scheduler.step(epoch + idx / len(train_loader))
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model_ema.eval()
        for batch in tqdm(val_loader):
            x, t = batch["data"], batch["target"]

            x = to_device(x, device)

            with torch.no_grad():
                y = model_ema(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            # print("save model")
            torch.save(model_ema.module.state_dict(), str(f"{output_path}/best_model_fold{fold}.pth"))

        elapsed_time = time() - epoch_start
        print(
            f"stage 1  [epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}sec")

        if epoch - best_epoch > CFG.es_patience:
            print("Early Stopping!")
            break

        train_loss = 0

    train_dataset = HMSHBASpecDataset(data2,
                                      specs=spectrograms,
                                      eeg_specs=all_eegs,
                                      aux_infos_by_group_id=aux_infos_by_group_id,
                                      aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                      transform=train_transform, phase='train')

    val_dataset = HMSHBASpecDataset(val2,
                                    specs=spectrograms,
                                    eeg_specs=all_eegs,
                                    aux_infos_by_group_id=aux_infos_by_group_id,
                                    aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                    transform=val_transform, phase='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_batch_size, num_workers=N_WORKERS, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=CFG.in_channels)

    model.load_state_dict(torch.load(str(f"{output_path}/best_model_fold{fold}.pth")), strict=True)
    model.to(device)
    model_ema = ModelEmaV3(model, decay=0.99, device=None, use_warmup=True, warmup_power=3 / 4)
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

    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time()
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]
            optimizer.zero_grad()
            with amp.autocast(use_amp, dtype=torch.bfloat16):
                y_0 = model(x)
                loss_0 = loss_func(y_0, t)

                loss = loss_0  # 0.5 * (loss_0 + loss_1) + alpha * loss_r

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema.update(model)
            if scheduler is not None:
                scheduler.step(epoch + idx / len(train_loader))
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model_ema.eval()
        for batch in tqdm(val_loader):
            x, t = batch["data"], batch["target"]

            x = to_device(x, device)

            with torch.no_grad():
                y = model_ema(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            # print("save model")
            torch.save(model_ema.module.state_dict(), str(f"{output_path}/best_model_fold{fold}.pth"))

        elapsed_time = time() - epoch_start
        print(
            f" stage 2 [epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}sec")

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


def tag_by_case(df):
    agg_dict = {**{m: 'first' for m in METAv1[:-2]}, **{t: 'sum' for t in CLASSES}}
    train = df.groupby('eeg_id').agg(agg_dict)

    train[CLASSES] = train[CLASSES] / train[CLASSES].values.sum(axis=1, keepdims=True)

    eeg_label_offset_seconds_min = df.groupby('eeg_id')[['eeg_label_offset_seconds']].min()
    eeg_label_offset_seconds_max = df.groupby('eeg_id')[['eeg_label_offset_seconds']].max()
    train['eeg_label_offset_seconds_min'] = eeg_label_offset_seconds_min.values
    train['eeg_label_offset_seconds_max'] = eeg_label_offset_seconds_max.values
    train['eeg_seconds'] = train['eeg_label_offset_seconds_max'] - train['eeg_label_offset_seconds_min'] + 50
    train['votes_sum_norm'] = train[CLASSES].values.max(axis=1)
    train = train.reset_index()
    ids_all_consistent = train[train.votes_sum_norm == 1.0].eeg_id.tolist()
    ids_multi_seg = train[train['eeg_seconds'] != 50].eeg_id.tolist()
    df['tag'] = ['case1'] * len(df)
    # tag if df eeg_id in ids_all_consistent but not in ids_multi_seg then tag 'case1'
    # not in ids_all_consistent, not in ids_multi_seg,  then tag 'case2'
    # in ids_all_consistent, in ids_multi_seg,  then tag 'case3'
    # not in ids_all_consistent, in ids_multi_seg,  then tag 'case4'
    cond_case1 = df['eeg_id'].isin(ids_all_consistent) & ~df['eeg_id'].isin(ids_multi_seg)
    cond_case2 = ~df['eeg_id'].isin(ids_all_consistent) & ~df['eeg_id'].isin(ids_multi_seg)
    cond_case3 = df['eeg_id'].isin(ids_all_consistent) & df['eeg_id'].isin(ids_multi_seg)

    cond_case4 = ~df['eeg_id'].isin(ids_all_consistent) & df['eeg_id'].isin(ids_multi_seg)

    # Apply conditions to 'tag' column
    df.loc[cond_case1, 'tag'] = 'case1'
    df.loc[cond_case2, 'tag'] = 'case2'
    df.loc[cond_case3, 'tag'] = 'case3'
    df.loc[cond_case4, 'tag'] = 'case4'

    print('sample level: ')
    print('total samples: ', len(df))
    print('1个段，意见相同: ', len(df[df.tag == 'case1']))
    print('1个段，意见不同: ', len(df[df.tag == 'case2']))

    print('多个段，意见相同: ', len(df[df.tag == 'case3']))
    print('多个段，意见不同: ', len(df[df.tag == 'case4']))

    print('eeg_id level: ')
    print('total eeg_ids: ', df.eeg_id.nunique())
    print('1个段，意见相同: ', df[df.tag == 'case1'].eeg_id.nunique())
    print('1个段，意见不同: ', df[df.tag == 'case2'].eeg_id.nunique())

    print('多个段，意见相同: ', df[df.tag == 'case3'].eeg_id.nunique())
    print('多个段，意见不同: ', df[df.tag == 'case4'].eeg_id.nunique())
    return df


def df_to_dict_by_eeg_id(df):
    """

    :return:
    {
      '1234': # eeg_id
      {
          'tag': 'case1',
          'eeg_label_offset_seconds_min': 0,
          'eeg_label_offset_seconds_max': 0,
          'spectrogram_label_offset_seconds_min': 0,
          'spectrogram_label_offset_seconds_max': 0,
          'n_segs': 2,
          'total_votes': 1000,
          'avg_targets': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],

          'subs': [
            {
               'eeg_label_offset_seconds': 0,
               'spectrogram_id': 353733,
               'spectrogram_label_offset_seconds': 0,
               'label_id': 127492639,
               'patient_id': 42516,
               'expert_consensus': 'Seizure',
               'votes': 10,
               #['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
               'targets': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

            },
            ..
          ]
      }
    }
    """
    eeg_ids = df.eeg_id.unique().tolist()
    infos = {}
    for eeg_id in eeg_ids:
        sub_df = df[df.eeg_id == eeg_id].copy()
        eeg_label_offset_seconds_min = sub_df['eeg_label_offset_seconds'].min()
        eeg_label_offset_seconds_max = sub_df['eeg_label_offset_seconds'].max()
        spectrogram_label_offset_seconds_min = sub_df['spectrogram_label_offset_seconds'].min()
        spectrogram_label_offset_seconds_max = sub_df['spectrogram_label_offset_seconds'].max()
        total_votes_cls6 = None
        item = {
            'tag': sub_df.iloc[0].tag,
            'eeg_label_offset_seconds_min': eeg_label_offset_seconds_min,
            'eeg_label_offset_seconds_max': eeg_label_offset_seconds_max,
            'spectrogram_label_offset_seconds_min': spectrogram_label_offset_seconds_min,
            'spectrogram_label_offset_seconds_max': spectrogram_label_offset_seconds_max,
            'n_segs': len(sub_df),
            'subs': []
        }
        for _, row in sub_df.iterrows():
            y = row[CLASSES].values
            if total_votes_cls6 is None:
                total_votes_cls6 = y
            else:
                total_votes_cls6 += np.array(y)
            n = y.sum()
            y = y / n
            d = {
                'eeg_label_offset_seconds': row['eeg_label_offset_seconds'],
                'spectrogram_id': row['spectrogram_id'],
                'spectrogram_label_offset_seconds': row['spectrogram_label_offset_seconds'],
                'label_id': row['label_id'],
                'patient_id': row['patient_id'],
                'expert_consensus': row['expert_consensus'],
                'votes': n,
                'targets': y
            }
            item['subs'].append(d)

        item['total_votes'] = total_votes_cls6.sum()
        item['avg_targets'] = total_votes_cls6 / total_votes_cls6.sum()
        infos[eeg_id] = item

    return infos


def preprocess_df(df):
    print('original df len: ', len(df))
    df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
    df2 = df2[df2['eeg_missing_ratio'] < 0.2]
    df = df2.copy().reset_index(drop=True)  # 在kaggle spec缺失率前提下eeg缺失率低的数据

    print('selected len by missing ratio', len(df))
    df['total_votes'] = df[CLASSES].values.sum(axis=1, keepdims=True)
    df[CLASSES] = df[CLASSES] / df[CLASSES].values.sum(axis=1, keepdims=True)

    # 分组1：按照eeg_id+targets进行分组
    grouped = df.groupby(['eeg_id',
                          'seizure_vote', 'lpg_vote', 'gpd_vote',
                          'irda_vote', 'grda_vote', 'other_vote'])

    print('groups: ', len(grouped))

    # 记录每个分组里投票数最多的信息
    all_max_votes_rows = []
    # 记录每组投票总数不统一情况的个数
    cnt = 0
    # 当前分组的id
    group_id = 0
    aux_infos_by_group_id = {}  # 记录每个eeg_id对应的多个子段的信息
    for name, group in grouped:
        # 在当前分组中找到拥有最大投票数行的索引
        idx = group['total_votes'].idxmax()
        # 获取最大投票数对应的行
        max_total_votes_row = group.loc[idx].copy()

        # 检查当前分组中的总投票数是否有多个不同的值
        if group['total_votes'].nunique() != 1:
            cnt += 1

        subs = []
        for _, row in group.iterrows():
            # 当前eeg_id对应每个子段的信息
            d = {
                'eeg_id': row['eeg_id'],
                'eeg_label_offset_seconds': row['eeg_label_offset_seconds'],
                'spectrogram_id': row['spectrogram_id'],
                'spectrogram_label_offset_seconds': row['spectrogram_label_offset_seconds'],
                'label_id': row['label_id'],
                'patient_id': row['patient_id'],
                'expert_consensus': row['expert_consensus'],
                'votes': row['total_votes'],  # 当前子段的总投票数
            }
            subs.append(d)
        aux_infos_by_group_id[group_id] = subs
        # 为拥有最大投票数的行数据添加一个新的group_id
        max_total_votes_row['group_id'] = grouped
        all_max_votes_rows.append(max_total_votes_row)
        group_id += 1

    # 记录一共有多少投票总数不一致组别的个数
    print('total_votes not all same groups: ', cnt)

    # Convert list of Series to DataFrame
    all_max_votes_rows_df = pd.concat(all_max_votes_rows, axis=1).T
    all_max_votes_rows_df = all_max_votes_rows_df.reset_index(drop=True)

    print('all_max_votes_rows len: ', len(all_max_votes_rows_df))
    print(all_max_votes_rows_df.iloc[1])
    aux_infos_by_eeg_id = df_to_dict_by_eeg_id(df)
    print('aux_infos_by_group_id len: ', len(aux_infos_by_group_id))
    print('aux_infos_by_eeg_id len: ', len(aux_infos_by_eeg_id))
    return all_max_votes_rows_df, aux_infos_by_group_id, aux_infos_by_eeg_id


def add_kl(data):
    labels = data[CLASSES].values + 1e-5

    # compute kl-loss with uniform distribution by pytorch
    data['kl'] = torch.nn.functional.kl_div(
        torch.log(torch.tensor(labels)),
        torch.tensor([1 / 6] * 6),
        reduction='none'
    ).sum(dim=1).numpy()
    return data


def main():
    train = pd.read_csv(f"{DATA}/train.csv")

    train = tag_by_case(train)

    train, aux_infos_by_group_id, aux_infos_by_eeg_id = preprocess_df(train)
    train = train.reset_index(drop=True)
    train = train[METAv2 + CLASSES]
    train.columns = ['eeg_id', 'spec_id', 'offset',
                     'patient_id', 'target', 'tag', 'total_votes',
                     'spectrogram_label_offset_seconds',
                     'eeg_label_offset_seconds', 'group_id'
                     ] + CLASSES

    train = add_kl(train)
    print(train.head(1).to_string())

    print('used dataset len: ', len(train))

    spectrograms = np.load(f"specs.npy", allow_pickle=True).item()
    all_eegs = np.load(CFG.chris_data, allow_pickle=True).item()
    # all_eegs_aug = np.load(f"eeg_specs_256x512_aug.npy", allow_pickle=True).item()
    score_list = []
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # gkf = GroupKFold(n_splits=N_FOLDS)
    for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
        output_path = CFG.exp_output_path
        print(f"[fold{i}]")
        score_list.append(
            train_one_fold(CFG, i, train_index, valid_index, train, spectrograms, all_eegs, output_path))
    print(score_list)

    # Inference Out of Fold
    N_rows = len(train)
    N_cols = len(CLASSES)
    all_oof = pd.DataFrame(np.zeros((N_rows, N_cols)), columns=CLASSES)
    all_true = pd.DataFrame(np.zeros((N_rows, N_cols)), columns=CLASSES)

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
            aux_infos_by_group_id=aux_infos_by_group_id,
            aux_infos_by_eeg_id=aux_infos_by_eeg_id,
            transform=val_transform, phase='val'
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=CFG.val_batch_size, num_workers=N_WORKERS, shuffle=False, drop_last=False)

        # get model
        model_path = f"{CFG.exp_output_path}/best_model_fold{i}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # inference
        val_pred = run_inference_loop(model, val_loader, device)
        # oof_pred_arr[valid_index] = val_pred

        all_oof.loc[valid_index, CLASSES] = val_pred
        all_true.loc[valid_index, CLASSES] = train.iloc[valid_index][CLASSES].values
        del valid_index
        del model, val_loader
        torch.cuda.empty_cache()
        gc.collect()

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
    oof.to_csv(f'{CFG.exp_output_path}/oof.csv', index=None)


if __name__ == "__main__":
    main()
