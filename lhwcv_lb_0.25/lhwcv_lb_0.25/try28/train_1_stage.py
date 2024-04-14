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
from raw_model3 import ECAPA_TDNN, HybridModel
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

save_dir = f'./v28_1_stage/{DATA_TYPE}/{args.model_type}_{args.backbone}_{args.secs}_{args.save_prefix}'

SEED = args.seed  # 42
setup_seed(SEED)
create_dir(save_dir)
os.system('cp *.py {}'.format(save_dir))

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
META = ['spectrogram_id', 'spectrogram_label_offset_seconds', 'patient_id', 'expert_consensus']

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


def preprocess_df2(df):
    print('origin df len: ', len(df))
    df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
    df2 = df2[df2['eeg_missing_ratio'] < 0.2]
    df = df2.copy().reset_index(drop=True)

    print('selected len by missing ratio: ', len(df))

    df['total_votes'] = df[TARGETS].values.sum(axis=1, keepdims=True)
    df[TARGETS] = df[TARGETS] / df[TARGETS].values.sum(axis=1, keepdims=True)

    grouped = df.groupby(['eeg_id'])

    print('groups: ', len(grouped))
    print('label ids: ',df['label_id'].nunique())
    cnt = 0

    wanted_label_ids = []
    for name, group in grouped:
        s = group.drop_duplicates(subset=["eeg_id"] + list(TARGETS))
        max_votes = group['total_votes'].max()
        if len(s) != 1:
            # print(group['total_votes'])
            # break
            cnt += 1
            min_votes = int(0.3*max_votes)
            group = group[group['total_votes'] > min_votes]
        wanted_label_ids.extend(group['label_id'].tolist())


    print('multi targets eegs: ', cnt)
    print('wanted_label_ids: ', len(wanted_label_ids))
    df = df[df['label_id'].isin(wanted_label_ids)].copy().reset_index(drop=True)
    print('final samples: ', len(df))
    print('final eegs: ', df['eeg_id'].nunique())
    return df


if not submission:
    train = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train_tag.csv')
    train = preprocess_df2(train)

    train = train.groupby('eeg_id')[META + TARGETS
                                    ].agg({**{m: 'first' for m in META}, **{t: 'sum' for t in TARGETS}}).reset_index()
    if add_reg:
        y_data = train[TARGETS].values + 0.166666667  # Regularization value

    train[TARGETS] = train[TARGETS] / train[TARGETS].values.sum(axis=1, keepdims=True)
    train.columns = ['eeg_id', 'spec_id', 'offset', 'patient_id', 'target'] + TARGETS

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
        if DATA_TYPE == 'both' or DATA_TYPE == 'eeg':
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

    if DATA_TYPE == 'raw' or DATA_TYPE == 'hybrid':
        LR = [1e-3, 1e-3, 1e-3, 1e-3,
              5e-4, 5e-4,
              1e-4, 1e-4,
              5e-5, 5e-5, 5e-5,
              1e-5]


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


from sklearn.model_selection import KFold, GroupKFold

if not submission:
    create_dir(save_dir)
    logger = TxtLogger(save_dir + "/logger.txt")
    logger.write('seed: {}'.format(SEED))
    # for CV scores setting random seed works for single GPU only
    setup_seed(SEED)
    all_oof_stage1 = []
    all_true = []
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
        # if i == 4:
        #     with_mixup = False
        #     augment_me = True
        # else:
        #     with_mixup = True
        #     augment_me = False

        train_gen = DataGenerator(data, augment=augment, specs=spectrograms,
                                  eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                  data_type=DATA_TYPE,
                                  random_common_reverse_signal=args.g_reverse_prob,
                                  random_common_negative_signal=args.g_neg_prob,
                                  random_reverse_signal=args.l_reverse_prob,
                                  random_negative_signal=args.l_neg_prob,
                                  secs=args.secs,
                                  )

        valid_gen = DataGenerator(val, mode='valid', specs=spectrograms,
                                  eeg_specs=all_eegs, raw_eegs=all_raw_eegs,
                                  data_type=DATA_TYPE,
                                  random_common_reverse_signal=args.g_reverse_prob,
                                  random_common_negative_signal=args.g_neg_prob,
                                  random_reverse_signal=args.l_reverse_prob,
                                  random_negative_signal=args.l_neg_prob,
                                  secs=args.secs,
                                  )

        BATCH_SIZE_PER_REPLICA = args.batch_size
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA

        train_dataset = DataLoader(train_gen, BATCH_SIZE, True,
                                   num_workers=6 if DATA_TYPE == 'raw' else 2)

        val_dataset = DataLoader(valid_gen, BATCH_SIZE, False,
                                 num_workers=4 if DATA_TYPE == 'raw' else 2)

        logger.write(f'### train size {len(train_gen)}, valid size {len(valid_gen)}')
        logger.write('#' * 25)

        model = build_model(args.backbone,
                            n_fft=args.n_fft,
                            win_length=args.win_n)
        #model.with_mixup = with_mixup

        save_prefix = '{}_fold_{}_stage1_'.format(DATA_TYPE, i + 1)
        oof_stage1, score_stage1 = fit_model(logger, model, LR,
                                             train_dataset, val_dataset,
                                             save_dir, save_prefix, len(LR),
                                             only_predict=False,
                                             weight_decay=0.0,
                                             early_stop_n=4,
                                             data_type=args.dtype)
        all_oof_stage1.append(oof_stage1)
        scores_stage1.append(score_stage1)

        del model, oof_stage1
        del train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

    all_oof_stage1 = np.concatenate(all_oof_stage1)
    all_true = np.concatenate(all_true)

    all_oof_stage1_df = pd.DataFrame(torch.softmax(torch.tensor(all_oof_stage1), dim=1))
    all_oof_stage1_df["id"] = np.arange(len(all_oof_stage1))
    y_true = pd.DataFrame(all_true)
    y_true["id"] = np.arange(len(all_true))
    stage1_score = score(solution=y_true, submission=all_oof_stage1_df, row_id_column_name="id")

    logger.write('#' * 25)
    #
    logger.write(f'CV KL SCORE stage1: {stage1_score}')
    logger.write(f'scores stage1: {scores_stage1}')

    logger.close()
