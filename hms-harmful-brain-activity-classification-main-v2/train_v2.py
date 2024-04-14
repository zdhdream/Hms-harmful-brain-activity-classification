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
#from gen_aux_info import preprocess_df

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

save_dir = f'./v28_0329_all/{DATA_TYPE}/{args.model_type}_{args.backbone}_{args.secs}_{args.t1}_{args.t2}{args.save_prefix}'



SEED = args.seed#42
setup_seed(SEED)
create_dir(save_dir)
os.system('cp *.py {}'.format(save_dir))

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
META = ['eeg_id', 'spectrogram_id', 'spectrogram_label_offset_seconds',
        'patient_id', 'expert_consensus', 'tag', 'total_votes',
        'spectrogram_label_offset_seconds',
        'eeg_label_offset_seconds',
        'group_id'
        ]

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
            y = row[TARGETS].values
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
    print('origin df len: ', len(df))
    df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
    df2 = df2[df2['eeg_missing_ratio'] < 0.2]
    df = df2.copy().reset_index(drop=True)

    print('selected len by missing ratio: ', len(df))

    df['total_votes'] = df[TARGETS].values.sum(axis=1, keepdims=True)
    df[TARGETS] = df[TARGETS] / df[TARGETS].values.sum(axis=1, keepdims=True)

    grouped = df.groupby(['eeg_id',
                          'seizure_vote', 'lpd_vote', 'gpd_vote',
                          'lrda_vote', 'grda_vote', 'other_vote'])

    print('groups: ', len(grouped))

    all_max_votes_rows = []
    cnt = 0
    group_id = 0
    aux_infos_by_group_id = {}
    for name, group in grouped:
        # print('len group: ', len(group))
        idx = group['total_votes'].idxmax()
        max_total_votes_row = group.loc[idx].copy()
        if group['total_votes'].nunique() != 1:
            # print(group['total_votes'])
            # break
            cnt += 1

        subs = []
        for _, row in group.iterrows():
            d = {
                'eeg_id': row['eeg_id'],
                'eeg_label_offset_seconds': row['eeg_label_offset_seconds'],
                'spectrogram_id': row['spectrogram_id'],
                'spectrogram_label_offset_seconds': row['spectrogram_label_offset_seconds'],
                'label_id': row['label_id'],
                'patient_id': row['patient_id'],
                'expert_consensus': row['expert_consensus'],
                'votes': row['total_votes'],
            }
            subs.append(d)
        aux_infos_by_group_id[group_id] = subs

        max_total_votes_row['group_id'] = group_id
        all_max_votes_rows.append(max_total_votes_row)
        group_id += 1

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

if not submission:
    train = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train_tag.csv')
    train, aux_infos_by_group_id, aux_infos_by_eeg_id = preprocess_df(train)
    train = train.reset_index(drop=True)
    train = train[META + TARGETS]
    train.columns = ['eeg_id', 'spec_id', 'offset',
                     'patient_id', 'target', 'tag', 'total_votes',
                     'spectrogram_label_offset_seconds',
                     'eeg_label_offset_seconds','group_id'
                     ] + TARGETS
    print('used dataset len: ', len(train))

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
            all_raw_eegs = np.load('../try4/eegs_all.npy',
                                   allow_pickle=True).item()
            # all_raw_eegs = np.load('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/eegs.npy',
            #                        allow_pickle=True).item()

from dataset_v2 import DataGenerator
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
        data2 = data2[data2['total_votes'] > args.t1].copy().reset_index()
        data2 = pd.concat([data2, data3])

        val1 = val[val['tag'].isin(['case1', 'case3'])].copy().reset_index()
        val2 = val[val['tag'].isin(['case2', 'case4'])].copy().reset_index()
        val3 = val1[val1['total_votes'] > args.t2].copy().reset_index()
        val2= val2[val2['total_votes'] > args.t1].copy().reset_index()
        val2 = pd.concat([val2, val3])

        #all_true_stage1.append(val1[TARGETS].values)
        #all_true_stage1.append(val2[TARGETS].values)
        #all_true_stage2.append(val2[TARGETS].values)

        logger.write('train len: {}'.format(len(data)))
        logger.write('stage1 train len:  {}'.format(len(data)))
        logger.write('stage2 train len: {}'.format(len(data2)))

        logger.write('stage2 val len: {}'.format(len(val)))
        logger.write('stage2 val len: {}'.format(len(val2)))
        #exit(0)

        train_gen = DataGenerator(data, augment=augment, specs=spectrograms,
                                  eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                  data_type=DATA_TYPE,
                                  aux_infos_by_group_id=aux_infos_by_group_id,
                                  aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                  random_common_reverse_signal=args.g_reverse_prob,
                                  random_common_negative_signal=args.g_neg_prob,
                                  random_reverse_signal=args.l_reverse_prob,
                                  random_negative_signal=args.l_neg_prob,
                                  secs=args.secs,
                                  )
        train_gen2 = DataGenerator(data2, augment=augment, specs=spectrograms,
                                   eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                   data_type=DATA_TYPE,
                                   aux_infos_by_group_id=aux_infos_by_group_id,
                                   aux_infos_by_eeg_id=aux_infos_by_eeg_id,
                                   random_common_reverse_signal=args.g_reverse_prob,
                                   random_common_negative_signal=args.g_neg_prob,
                                   random_reverse_signal=args.l_reverse_prob,
                                   random_negative_signal=args.l_neg_prob,
                                   secs=args.secs,
                                   )

        valid_gen1 = DataGenerator(val, mode='valid', specs=spectrograms,
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
                                             train_dataset, val_dataset1,                              save_dir, save_prefix, len(LR),
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
