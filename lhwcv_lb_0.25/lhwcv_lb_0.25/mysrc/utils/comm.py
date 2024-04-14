import  numpy as np
import os
import errno
import  random
import  torch
import pandas as pd

def setup_seed(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = deterministic


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def get_df_by_flist_str(flist_str, exclude_ke_0_17=False):
    csvs = flist_str.split(',')
    df = pd.read_csv(csvs[0])
    if 'KeSpeech_' in csvs[0] and exclude_ke_0_17:
        exclude_classes_list = list(range(0, 18))
        df = df[~df['label'].isin(exclude_classes_list)]

        #df.to_csv('./data/external/KeSpeech_VAD_slices_8k_stage2_18_41.csv',encoding='utf-8', index=False)
        #exit(0)

    for i in range(1, len(csvs)):
        df2 = pd.read_csv(csvs[i])
        if 'KeSpeech_' in csvs[i] and exclude_ke_0_17:
            exclude_classes_list = list(range(0, 18))
            df2 = df2[~df['label'].isin(exclude_classes_list)]

        df = pd.concat([df, df2])

    return df


def gen_idx_by_sliding_window(volume_size, patch_size, shift_size):
    D, H, W = volume_size
    pD, pH, pW = patch_size
    sD, sH, sW = shift_size
    # gen a list of voxel idx
    # eg idx0:  [(0, pD), (0, pH), (0, pW)]
    #    idx1:  [(0, pD), (0, pH), (sW, sW + pW) ]
    #    ...
    idx = []
    for z in range(0, D, sD):
        for y in range(0, H, sH):
            for x in range(0, W, sW):
                # Prefix 'i' stands for 'initial' and 'f' stands for 'final'
                idx_zi, idx_zf = z, min(z + pD, D)
                idx_yi, idx_yf = y, min(y + pH, H)
                idx_xi, idx_xf = x, min(x + pW, W)

                # accounts for cases where the patches have to be modified near the border
                if idx_zf == D:
                    idx_zi = D - pD
                if idx_yf == H:
                    idx_yi = H - pH
                if idx_xf == W:
                    idx_xi = W - pW

                idx.append([(idx_zi, idx_zf), (idx_yi, idx_yf), (idx_xi, idx_xf)])

    return idx