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
    df['tag'] = ['case1']*len(df)
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
    print('1个段，意见相同: ',  df[df.tag == 'case1'].eeg_id.nunique())
    print('1个段，意见不同: ',  df[df.tag == 'case2'].eeg_id.nunique())

    print('多个段，意见相同: ', df[df.tag == 'case3'].eeg_id.nunique())
    print('多个段，意见不同: ', df[df.tag == 'case4'].eeg_id.nunique())
    #print(df.iloc[0])
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

df = pd.read_csv('/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/train.csv')
df = tag_by_case(df)
infos = df_to_dict_by_eeg_id(df)
import pickle
save_name = '/home/hw/m2_disk/kaggle/data/hms-harmful-brain-activity-classification/aux_infos.pkl'
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
