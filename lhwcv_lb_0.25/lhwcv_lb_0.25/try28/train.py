# -*- coding: utf-8 -*-
import os, random
import torch
import gc
from torch.utils.data import DataLoader
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from mysrc.utils.comm import setup_seed, create_dir
from mysrc.utils.logger import TxtLogger
from kaggle_kl_div import score
import argparse
from raw_model import EEGNet
from raw_model2 import DilatedInceptionWaveNet
from raw_model3 import ECAPA_TDNN,HybridModel
from raw_model4 import ModelCWT

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--dtype', type=str, help='data_type', default='kaggle')
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--save_prefix', type=str, default='')
parser.add_argument('--add_reg', type=int, default=0)
parser.add_argument('--backbone', type=str, default='b2')
parser.add_argument('--g_reverse_prob', type=float, default=0.0)
parser.add_argument('--l_reverse_prob', type=float, default=0.0)
parser.add_argument('--g_neg_prob', type=float, default=0.0)
parser.add_argument('--l_neg_prob', type=float, default=0.0)
parser.add_argument('--secs', type=int, default=50)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--win_n', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_type', type=str, default='raw_spec')
parser.add_argument('--t1', type=int, default=10)
parser.add_argument('--t2', type=int, default=5)

args = parser.parse_args()
print(args)
VER = 47
# DATA_TYPE = 'kaggle'  # both|eeg|kaggle|raw
DATA_TYPE = args.dtype
TEST_MODE = False
submission = False
augment = True if args.aug == 1 else False
add_reg = True if args.add_reg == 1 else False
MODEL_TYPE = args.model_type
# save_dir = './v28/wkdir_{}_{}_{}_sec{}_win{}_t1_{}_t2_{}_seed_{}{}/'.format(DATA_TYPE,
#                                                              args.model_type,
#                                                              args.backbone,
#                                                               args.secs,
#                                                               args.win_n,
#                                                               args.t1, args.t2,
#                                                               args.seed,
#                                                               args.save_prefix)

save_dir = f'./v28_0326/{DATA_TYPE}/{args.model_type}_{args.backbone}_{args.secs}_{args.t1}_{args.t2}{args.save_prefix}'


# Setup for ensemble
ENSEMBLE = True
LBs = [0.37, 0.39, 0.41, 0.41]  # for weighted ensemble we use LBs of each model
VERK = 43  # Kaggle's spectrogram model version
VERB = 47  # Kaggle's and EEG's spectrogram model version
VERE = 42  # EEG's spectrogram model version
VERR = 37  # EEG's raw wavenet model version, trained on single GPU

SEED = args.seed#42
setup_seed(SEED)
create_dir(save_dir)
os.system('cp *.py {}'.format(save_dir))

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
META = ['spectrogram_id', 'spectrogram_label_offset_seconds', 'patient_id', 'expert_consensus', 'tag', 'total_votes']

# FEATS2 = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
# FEAT2IDX = {x: y for x, y in zip(FEATS2, range(len(FEATS2)))}
FEATS2 = ["Fp1", "T3", "C3", "O1", "F7", "T5", "F3", "P3",
          "Fp2", "C4", "T4", "O2", "F8", "T6", "F4", "P4"]

FEAT2IDX = {x: y for x, y in zip(FEATS2, range(len(FEATS2)))}


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset:offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
    for j, col in enumerate(FEATS2):

        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        data[:, j] = x

    return data


def add_kl(data):
    import torch
    labels = data[TARGETS].values + 1e-5

    # compute kl-loss with uniform distribution by pytorch
    data['kl'] = torch.nn.functional.kl_div(
        torch.log(torch.tensor(labels)),
        torch.tensor([1 / 6] * 6),
        reduction='none'
    ).sum(dim=1).numpy()
    return data


def tag_by_case(df):
    agg_dict = {**{m: 'first' for m in META[:-2]}, **{t: 'sum' for t in TARGETS}}
    train = df.groupby('eeg_id').agg(agg_dict)

    train[TARGETS] = train[TARGETS] / train[TARGETS].values.sum(axis=1, keepdims=True)

    eeg_label_offset_seconds_min = df.groupby('eeg_id')[['eeg_label_offset_seconds']].min()
    eeg_label_offset_seconds_max = df.groupby('eeg_id')[['eeg_label_offset_seconds']].max()
    train['eeg_label_offset_seconds_min'] = eeg_label_offset_seconds_min.values
    train['eeg_label_offset_seconds_max'] = eeg_label_offset_seconds_max.values
    train['eeg_seconds'] = train['eeg_label_offset_seconds_max'] - train['eeg_label_offset_seconds_min'] + 50
    train['votes_sum_norm'] = train[TARGETS].values.max(axis=1)
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
    # print(df.iloc[0])
    df['total_votes'] = df[TARGETS].values.sum(axis=1, keepdims=True)

    case1_3_df = df[df.tag.isin(['case1', 'case3'])].copy().reset_index()
    case2_df = df[df.tag == 'case2'].copy().reset_index()
    case4_df = df[df.tag == 'case4'].copy().reset_index()

    case2_df = case2_df[case2_df['total_votes'] >= args.t1].copy().reset_index()
    case4_df = case4_df[case4_df['total_votes'] >= args.t1].copy().reset_index()

    df = pd.concat([case1_3_df, case2_df, case4_df])
    print('eeg_id level2 after filter: ')
    print('total eeg_ids: ', df.eeg_id.nunique())
    print('1个段，意见相同: ', df[df.tag == 'case1'].eeg_id.nunique())
    print('1个段，意见不同: ', df[df.tag == 'case2'].eeg_id.nunique())

    print('多个段，意见相同: ', df[df.tag == 'case3'].eeg_id.nunique())
    print('多个段，意见不同: ', df[df.tag == 'case4'].eeg_id.nunique())
    # exit(0)
    return df


if not submission:
    train = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train.csv')
    train = tag_by_case(train)
    # train = train[train['tag'].isin(['case1', 'case2', 'case3'])].copy().reset_index()
    print(train.iloc[0])
    train = train.groupby('eeg_id')[META + TARGETS
                                    ].agg({**{m: 'first' for m in META}, **{t: 'sum' for t in TARGETS}}).reset_index()
    if add_reg:
        y_data = train[TARGETS].values + 0.166666667  # Regularization value

    train[TARGETS] = train[TARGETS] / train[TARGETS].values.sum(axis=1, keepdims=True)
    train.columns = ['eeg_id', 'spec_id', 'offset', 'patient_id', 'target', 'tag', 'total_votes'] + TARGETS
    train = add_kl(train)
    print(train.head(1).to_string())

    # train['votes_sum_norm'] = train[TARGETS].values.max(axis=1)
    # ids = train[train.votes_sum_norm == 1.0].eeg_id.tolist()
    print('eeg_ids len: ', len(train))
    # print('train.votes_sum_norm == 1.0: ', len(ids))
    # exit(0)
    if add_reg:
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

if not submission:
    # FOR TESTING SET TEST_MODE TO TRUE
    if TEST_MODE:
        train = train.sample(500, random_state=SEED).reset_index(drop=True)
        spectrograms = {}
        for i, e in enumerate(train.spec_id.values):
            if i % 100 == 0: print(i, ', ', end='')
            x = pd.read_parquet(
                f'/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/{e}.parquet')
            spectrograms[e] = x.values
        all_eegs = {}
        for i, e in enumerate(train.eeg_id.values):
            if i % 100 == 0: print(i, ', ', end='')
            x = np.load(f'/kaggle/input/eeg-spectrograms/EEG_Spectrograms/{e}.npy')
            all_eegs[e] = x
        all_raw_eegs = {}
        for i, e in enumerate(train.eeg_id.values):
            if i % 100 == 0: print(i, ', ', end='')
            x = eeg_from_parquet(f'/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{e}.parquet')
            all_raw_eegs[e] = x
    else:
        spectrograms = None
        all_eegs = None
        all_raw_eegs = None
        if DATA_TYPE == 'both' or DATA_TYPE == 'kaggle' or DATA_TYPE == 'hybrid':
            spectrograms = np.load(
                '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/kaggle_specs.npy',
                allow_pickle=True).item()
        if DATA_TYPE == 'both' or DATA_TYPE == 'eeg' or DATA_TYPE == 'hybrid':
            all_eegs = np.load(
                '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/eeg_specs_v2.npy',
                allow_pickle=True).item()
        if DATA_TYPE == 'raw' or DATA_TYPE == 'hybrid':
            all_raw_eegs = np.load('../try4/eegs.npy',
                                   allow_pickle=True).item()
            # all_raw_eegs = np.load('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/eegs.npy',
            #                        allow_pickle=True).item()

from dataset import DataGenerator
from model import MyModel
from trainer import KLDivWithLogitsLoss, fit_model

if not submission:

    LR = [1e-3, 1e-3, 5e-4,
          1e-4, 1e-4, 1e-4,
          5e-5]
    # LR2 = [1e-5, 1e-5, 1e-6]
    LR2 = [1e-4, 1e-4, 5e-5, 1e-5, 1e-5, 1e-6]

    if DATA_TYPE == 'raw' or DATA_TYPE == 'hybrid':
        LR = [1e-3, 1e-3, 1e-3, 1e-3,
              5e-4, 5e-4,
              1e-4, 1e-4,
              5e-5,
              1e-5]
        # LR = [5e-4, 2-4, 2e-4, 1e-4,
        #       1e-4, 1e-4,
        #       1e-4, 1e-4,
        #       5e-5,
        #       1e-5]

        #LR2 = [1e-4, 1e-4, 1e-5, 1e-5, 1e-6]
        #LR2 = [1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5, 1e-6]
        LR2 = [1e-4, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5, 1e-5, 1e-5, 1e-6]


    # if DATA_TYPE == 'hybrid':
    #     LR = [1e-3, 1e-3, 1e-3,
    #           5e-4, 5e-4,
    #           1e-4,
    #           5e-5,
    #           1e-5]
    #     LR2 = [1e-5, 1e-5, 1e-6]

import torch.nn as nn


# class HybridModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #self.raw_model = DilatedInceptionWaveNet()
#         self.raw_model = ECAPA_TDNN()
#         self.spec_model = MyModel()
#         self.proj = nn.Linear(12, 6)
#
#     def forward(self, x1, x2, targets=None):
#         fea1 = self.spec_model(x1)
#         if targets is not None:
#             fea2, targets = self.raw_model(x2, targets)
#         else:
#             fea2 = self.raw_model(x2)
#         fea = torch.cat((fea1, fea2), dim=1)
#         out = self.proj(fea)
#         if self.training and targets is not None:
#             return out, targets
#         return out


def build_model(backbone,
                n_fft=1024,
                win_length=128,
                num_mels=128,
                width=300):
    if MODEL_TYPE == 'hybrid':
        return HybridModel(backbone,
                          n_fft, win_length, num_mels, width).cuda()
    if MODEL_TYPE == 'raw_spec':
        return ECAPA_TDNN(backbone,
                          n_fft, win_length, num_mels, width).cuda()
    if MODEL_TYPE == 'cwt':
        return ModelCWT(backbone,
                          n_fft, win_length, num_mels, width).cuda()
    if MODEL_TYPE == 'wavenet':
        return DilatedInceptionWaveNet().cuda()
    return MyModel(backbone).cuda()


# def score(y_pred, y_true):
#     y_pred = torch.from_numpy(y_pred).float()
#     #y_pred = torch.softmax(torch.tensor(y_pred), dim=1)
#     print('y_pred shape: ', y_pred.shape)
#     y_true = torch.from_numpy(y_true).float()
#     print('y_pred mean: ', y_pred.mean())
#     print('y_true mean: ', y_true.mean())
#     print('y_true std: ', y_true.std())
#
#     kl = KLDivWithLogitsLoss()
#     return kl(y_pred, y_true)


from sklearn.model_selection import KFold, GroupKFold

if not submission:
    create_dir(save_dir)
    logger = TxtLogger(save_dir + "/logger.txt")
    logger.write('seed: {}'.format(SEED))
    # for CV scores setting random seed works for single GPU only
    setup_seed(SEED)
    all_oof_stage1 = []
    all_oof_stage2 = []
    all_true = []
    all_true_stage1 = []
    all_true_stage2 = []
    losses = []
    val_losses = []
    total_hist = {}
    scores_stage1 = []
    scores_stage2 = []
    gkf = GroupKFold(n_splits=5)
    all_datas = []
    for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
        data, val = train.iloc[train_index].copy(), train.iloc[valid_index].copy()
        data = data.reset_index(drop=True)
        val = val.reset_index(drop=True)
        all_datas.append((data, val))

    for i in range(5):
        logger.write('#' * 25)
        logger.write(f'### Fold {i + 1}')
        setup_seed(SEED)

        data, val = all_datas[i]
        all_true.append(val[TARGETS].values)

        data1 = data[data['tag'].isin(['case1', 'case3'])].copy().reset_index()
        data2 = data[data['tag'].isin(['case2', 'case4'])].copy().reset_index()
        data3 = data1[data1['total_votes'] > args.t2].copy().reset_index()
        print('data3 len: ', len(data3))
        data2 = pd.concat([data2, data3])

        val1 = val[val['tag'].isin(['case1', 'case3'])].copy().reset_index()
        val2 = val[val['tag'].isin(['case2', 'case4'])].copy().reset_index()
        val3 = val1[val1['total_votes'] > args.t2].copy().reset_index()
        val2 = pd.concat([val2, val3])

        #all_true_stage1.append(val1[TARGETS].values)
        all_true_stage1.append(val2[TARGETS].values)
        all_true_stage2.append(val2[TARGETS].values)

        print('train len: ', len(data))
        print('stage1 train len: ', len(data1))
        print('stage2 train len: ', len(data2))

        train_gen = DataGenerator(data1, augment=augment, specs=spectrograms,
                                  eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                  data_type=DATA_TYPE,
                                  random_common_reverse_signal=args.g_reverse_prob,
                                  random_common_negative_signal=args.g_neg_prob,
                                  random_reverse_signal=args.l_reverse_prob,
                                  random_negative_signal=args.l_neg_prob,
                                  secs=args.secs,
                                  )
        train_gen2 = DataGenerator(data2, augment=augment, specs=spectrograms,
                                   eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                   data_type=DATA_TYPE,
                                   random_common_reverse_signal=args.g_reverse_prob,
                                   random_common_negative_signal=args.g_neg_prob,
                                   random_reverse_signal=args.l_reverse_prob,
                                   random_negative_signal=args.l_neg_prob,
                                   secs=args.secs,
                                   )

        valid_gen1 = DataGenerator(val1, mode='valid', specs=spectrograms,
                                   eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                   data_type=DATA_TYPE,
                                   random_common_reverse_signal=args.g_reverse_prob,
                                   random_common_negative_signal=args.g_neg_prob,
                                   random_reverse_signal=args.l_reverse_prob,
                                   random_negative_signal=args.l_neg_prob,
                                   secs=args.secs,
                                   )
        # data = data[data['kl'] < 5.5]

        valid_gen2 = DataGenerator(val2, mode='valid', specs=spectrograms,
                                   eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                   data_type=DATA_TYPE,
                                   random_common_reverse_signal=args.g_reverse_prob,
                                   random_common_negative_signal=args.g_neg_prob,
                                   random_reverse_signal=args.l_reverse_prob,
                                   random_negative_signal=args.l_neg_prob,
                                   secs=args.secs,
                                   )
        # in_shape = (2000, 8) if DATA_TYPE == 'raw' else (512, 512, 3)
        EPOCHS = 5
        BATCH_SIZE_PER_REPLICA = args.batch_size#32 if DATA_TYPE == 'raw' else 16
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA

        train_dataset = DataLoader(train_gen, BATCH_SIZE, True,
                                   num_workers=6 if DATA_TYPE == 'raw' else 2)
        train_dataset2 = DataLoader(train_gen2, BATCH_SIZE, True,
                                    num_workers=2)

        val_dataset1 = DataLoader(valid_gen1, BATCH_SIZE, False,
                                  num_workers=4 if DATA_TYPE == 'raw' else 2)
        val_dataset2 = DataLoader(valid_gen2, BATCH_SIZE, False,
                                  num_workers=4 if DATA_TYPE == 'raw' else 2)

        logger.write(f'### train size {len(train_gen)}, valid size {len(valid_gen1)}')
        logger.write('#' * 25)

        model = build_model(args.backbone,
                            n_fft=args.n_fft,
                            win_length=args.win_n)
        # if i != 1:
        #     continue
        save_prefix = '{}_fold_{}_stage1_'.format(DATA_TYPE, i + 1)
        oof_stage1, score_stage1 = fit_model(logger, model, LR,
                                             train_dataset, val_dataset2,
                                             save_dir, save_prefix, len(LR),
                                             only_predict=False,
                                             weight_decay=1e-5,
                                             early_stop_n=3,
                                             data_type=args.dtype)
        all_oof_stage1.append(oof_stage1)
        scores_stage1.append(score_stage1)

        logger.write(f'### seconds stage train size {len(data2)}, valid size {len(val2)}')
        logger.write('#' * 25)
        save_prefix = '{}_fold_{}_stage2_'.format(DATA_TYPE, i + 1)
        model.with_mixup = False
        oof_stage2, score_stage2 = fit_model(logger, model, LR2,
                                             train_dataset2, val_dataset2,
                                             save_dir, save_prefix, len(LR2),
                                             only_predict=False, weight_decay=0.0,
                                             early_stop_n=5,
                                             data_type=args.dtype
                                             )
        all_oof_stage2.append(oof_stage2)
        scores_stage2.append(score_stage2)

        del model, oof_stage1, oof_stage2
        del train_dataset, train_dataset2, val_dataset1, val_dataset2
        gc.collect()
        torch.cuda.empty_cache()

    all_oof_stage1 = np.concatenate(all_oof_stage1)
    all_oof_stage2 = np.concatenate(all_oof_stage2)
    all_true = np.concatenate(all_true)
    all_true_stage1 = np.concatenate(all_true_stage1)
    all_true_stage2 = np.concatenate(all_true_stage2)

    all_oof_stage1_df = pd.DataFrame(torch.softmax(torch.tensor(all_oof_stage1), dim=1))
    all_oof_stage1_df["id"] = np.arange(len(all_oof_stage1))
    y_true = pd.DataFrame(all_true_stage1)
    y_true["id"] = np.arange(len(all_true_stage1))
    stage1_score = score(solution=y_true, submission=all_oof_stage1_df, row_id_column_name="id")

    all_oof_stage2_df = pd.DataFrame(torch.softmax(torch.tensor(all_oof_stage2), dim=1))
    all_oof_stage2_df["id"] = np.arange(len(all_oof_stage2))
    y_true = pd.DataFrame(all_true_stage2)
    y_true["id"] = np.arange(len(all_true_stage2))
    stage2_score = score(solution=y_true, submission=all_oof_stage2_df, row_id_column_name="id")

    logger.write('#' * 25)
    #
    logger.write(f'CV KL SCORE stage1: {stage1_score}')
    logger.write(f'CV KL SCORE stage2: {stage2_score}')

    logger.write(f'scores stage1: {scores_stage1}')
    logger.write(f'scores stage2: {scores_stage2}')

    logger.close()
