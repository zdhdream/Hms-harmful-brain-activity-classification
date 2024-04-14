# -*- coding: utf-8 -*-

"""
case 1:
1个段且意见相同
case 2:
1个段但意见不同
case 3:
意见相同，且多个段
case 4:
意见不同，且多个段
"""

import numpy as np
import pandas as pd
import tqdm

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
META = ['spectrogram_id', 'spectrogram_label_offset_seconds', 'patient_id', 'expert_consensus']


def tag_by_case(df):
    agg_dict = {**{m: 'first' for m in META}, **{t: 'sum' for t in TARGETS}}
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
    for eeg_id in tqdm.tqdm(eeg_ids):
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


DATA_ROOT = '/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/'
feats = [
    "Fp1", "T3", "C3", "O1", "F7", "T5", "F3", "P3",
    "Fp2", "C4", "T4", "O2", "F8", "T6", "F4", "P4",
    # "Fz", "Cz", "Pz"
]


def tag_kaggle_spec_missing_ratio(df):
    print('tag_kaggle_spec_missing_ratio..')
    spectrograms = np.load(
        f'{DATA_ROOT}/kaggle_specs.npy',
        allow_pickle=True).item()
    for idx, row in tqdm.tqdm(df.iterrows()):
        spec = spectrograms[row.spectrogram_id]
        offset = int(row['spectrogram_label_offset_seconds'] // 2)
        spec = pd.DataFrame(spec)
        df.loc[idx, 'kaggle_spec_missing_ratio'] = spec.iloc[offset: 300 + offset, 1:].isna().mean().mean()
    return df


def tag_eeg_missing_ratio(df):
    print('tag_eeg_missing_ratio..')
    for eeg_id, dff in tqdm.tqdm(df.groupby('eeg_id')):
        eeg = pd.read_parquet(f'{DATA_ROOT}/train_eegs/{eeg_id}.parquet')
        for idx, row in dff.iterrows():
            offset = int(row['eeg_label_offset_seconds']) * 200
            df.loc[idx, 'eeg_missing_ratio'] = eeg.iloc[offset: offset + 10000, 1:].isna().mean().mean()
    return df


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


def preprocess_df2(df):
    print('origin df len: ', len(df))
    # df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
    # df2 = df2[df2['eeg_missing_ratio'] < 0.2]
    # df = df2.copy().reset_index(drop=True)

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

# df = pd.read_csv(f'{DATA_ROOT}/train.csv')
# df = tag_by_case(df)
# df = tag_kaggle_spec_missing_ratio(df)
# df = tag_eeg_missing_ratio(df)
#
# df.to_csv(f'{DATA_ROOT}/train_tag.csv')
# exit(0)

df = pd.read_csv(f'{DATA_ROOT}/train_tag.csv')
preprocess_df2(df)
exit(0)

# import matplotlib.pyplot as plt
#
# n_samples = len(df)
# df0 = df[df['kaggle_spec_missing_ratio'] > 0]
# df1 = df[df['eeg_missing_ratio'] > 0]
#
# df2 = df[df['kaggle_spec_missing_ratio'] < 0.5]
# df2 = df2[df2['eeg_missing_ratio'] < 0.2]
# print('df len: ', len(df))
# print('df2 len: ', len(df2))
# print('no nan eeg_ids: ', len(df2.eeg_id.unique()))
# exit(0)
#
# print(df0['kaggle_spec_missing_ratio'].describe())
# print(df1['eeg_missing_ratio'].describe())
#
# print('kaggle missing ratio: ', len(df0) / n_samples)
# print('eeg missing ratio: ', len(df1) / n_samples)
# df0['kaggle_spec_missing_ratio'].plot.hist(bins=30,
#                                            title='kaggle_spec_missing_ratio freq', xlabel='Portion of Missing Data')
# plt.show()
# plt.figure()
# df1['eeg_missing_ratio'].plot.hist(bins=30,
#                                    title='eeg_missing_ratio freq', xlabel='Portion of Missing Data')
# plt.show()
# exit(0)



infos = df_to_dict_by_eeg_id(df)
import pickle

save_name = '/home/lhw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/aux_infos.pkl'
pickle.dump(infos, open(save_name, 'wb'))
print('saved to: ', save_name)
for k, v in infos.items():
    print('eeg_id: ', k)
    print(v)
    break
exit(0)
agg_dict = {**{m: 'first' for m in META}, **{t: 'sum' for t in TARGETS}}
train = df.groupby('eeg_id').agg(agg_dict)

train[TARGETS] = train[TARGETS] / train[TARGETS].values.sum(axis=1, keepdims=True)

eeg_label_offset_seconds_min = df.groupby('eeg_id')[['eeg_label_offset_seconds']].min()
eeg_label_offset_seconds_max = df.groupby('eeg_id')[['eeg_label_offset_seconds']].max()
train['eeg_label_offset_seconds_min'] = eeg_label_offset_seconds_min.values
train['eeg_label_offset_seconds_max'] = eeg_label_offset_seconds_max.values
train['eeg_seconds'] = train['eeg_label_offset_seconds_max'] - train['eeg_label_offset_seconds_min'] + 50
train['votes_sum_norm'] = train[TARGETS].values.max(axis=1)
train = train.reset_index()
print('total: ', len(train))

print('start is not 0: ', len(train[train['eeg_label_offset_seconds_min'] != 0]))

print('only 50s from start: ', len(train[train['eeg_label_offset_seconds_max'] == 0]))

print(train['eeg_seconds'].describe())

# ids = train[train.eeg_seconds > 2000].eeg_id.tolist()
# print(ids)
# print('eegs: ', len(ids))
# d1 = df[df.eeg_id.isin(ids)]
#
# d1 = d1.drop_duplicates(subset=["eeg_id"] + list(TARGETS))
# print('d1 len: ', len(d1))
# print(train[train.eeg_seconds == 3422].iloc[0])
# d0 = df[df.eeg_id == 188361788].copy().reset_index()
# d0['total_votes'] = d0[TARGETS].values.sum(axis=1, keepdims=True)
#
# print(d0.iloc[0])
#
# print(d0['expert_consensus'].describe())

train['votes_sum_norm'] = train[TARGETS].values.max(axis=1)
ids = train[train.votes_sum_norm == 1.0].eeg_id.tolist()
print('train.votes_sum_norm == 1.0: ', len(ids))
d1 = df[df.eeg_id.isin(ids)].copy().reset_index()
print('one type only len: ', len(d1))
d1['total_votes'] = d1[TARGETS].values.sum(axis=1, keepdims=True)
d1[TARGETS] = d1[TARGETS] / d1[TARGETS].values.sum(axis=1, keepdims=True)
print('d1 len: ', len(d1))
d1_uni = d1.drop_duplicates(subset=["eeg_id"] + list(TARGETS))
print('d1_uni len: ', len(d1_uni))
assert len(d1_uni) == len(ids)

print(d1['total_votes'].describe())

train1 = train[train.eeg_id.isin(ids)].copy().reset_index()
print(train1['eeg_seconds'].describe())

print('多个段,且意见相同：', len(train1[train1['eeg_seconds'] != 50]))

train2 = train[train['eeg_label_offset_seconds_max'] == 0].copy().reset_index()
print('1个段：', len(train2))
print('1个段, 且意见相同：', len(train2[train2.eeg_id.isin(ids)]))
print('1个段, 意见不同：', len(train2[~train2.eeg_id.isin(ids)]))
