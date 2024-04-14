import random

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from cfg import CFG

CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)


class HMSHBASpecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
            specs,
            eeg_specs,
            aux_infos_by_group_id,
            aux_infos_by_eeg_id,
            transform: A.Compose,
            phase: str
    ):
        self.data = data
        self.specs = specs  # kaggle spec
        self.eeg_specs = eeg_specs  # eeg spec
        self.aux_infos_by_group_id = aux_infos_by_group_id
        self.aux_infos_by_eeg_id = aux_infos_by_eeg_id
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5

    def __len__(self):
        return len(self.data)

    def smooth_labels(self, labels, factor=0.01):
        labels *= (1 - factor)
        labels += (factor / 6)
        return labels

    def make_Xy(self, new_ind):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        # img = np.ones((512, 512), dtype="float32")

        row = self.data.iloc[new_ind]

        if self.phase == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        img = self.specs[row.spec_id][:, :].T  # (256, 400 or ???)

        if  self.phase == "train":

            if random.random() < 0.5:
                group = self.aux_infos_by_group_id[row.group_id]
                item = random.sample(group, 1)[0]
                img = self.specs[item['spectrogram_id']].T
                offset = int(item['spectrogram_label_offset_seconds']) // 2
            else:
                offset = int(row.spectrogram_label_offset_seconds) // 2

        else:
            offset = int(row.spectrogram_label_offset_seconds) // 2

        # print(row.spec_id, img.shape)
        ch = img.shape[1] // 2
        if ch >= 256:
            img = self.specs[row.spec_id][ch - 256:ch + 256, :].T  # (256, 512)
        else:
            img = self.specs[row.spec_id][:, :].T  # (256, ???)

        # print(row.spec_id, img.shape)
        h, w = img.shape[:2]

        # if w!=256:
        #    print(row.spec_id, img.shape)

        # log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        # crop to 256 time steps
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (256, 512, 4)

        X[0:256, :, 1] = img[:, :, 0]  # (256, 512, 5)
        X[256:512, :, 1] = img[:, :, 1]  # (256, 512, 5)
        X[0:256, :, 2] = img[:, :, 2]  # (256, 512, 5)
        X[256:512, :, 2] = img[:, :, 3]  # (256, 512, 5)

        if self.phase == 'train':
            X = self.spec_mask(X)

            if torch.rand(1) > self.aug_prob:
                X = self.shift_img(X)

        X = self._apply_transform(X)

        X = [X[i:i + 1, :, :] for i in range(3)]  # x: [3,512,512]
        X = torch.cat(X, dim=1)  # (1, 1536, 512)

        y[:] = row[CLASSES]

        # if self.phase =='train':
        #      y[:] = self.smooth_labels(row[CLASSES])

        return X, y

    def __getitem__(self, index: int):

        X1, y1 = self.make_Xy(index)

        if torch.rand(1) > 0.5 and self.phase == 'train':
            index2 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X2, y2 = self.make_Xy(index2)

        else:
            X2, y2 = X1, y1

        if torch.rand(1) > 0.5 and self.phase == 'train':
            index3 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X3, y3 = self.make_Xy(index3)

        else:
            X3, y3 = X1, y1

        X = torch.cat([X1, X2, X3], dim=0)  # (3, 1536, 512)
        y = (y1 + y2 + y3) / 3

        return X, y

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img

    def shift_img(self, img):
        s = torch.randint(0, CFG.IMG_SIZE[1], (1,))[0]
        new = np.concatenate([img[:, s:], img[:, :s]], axis=1)
        return new

    def spec_mask(self, img, max_it=4):
        count = 0
        new = img
        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[0] - CFG.IMG_SIZE[0] // 16, (1,))[0]
            h = torch.randint(CFG.IMG_SIZE[0] // 32, CFG.IMG_SIZE[0] // 16, (1,))[0]
            new[s:s + h] *= 0
            count += 1

        count = 0

        while count < max_it and torch.rand(1) > self.aug_prob:
            s = torch.randint(0, CFG.IMG_SIZE[1] - CFG.IMG_SIZE[1] // 16, (1,))[0]
            w = torch.randint(CFG.IMG_SIZE[1] // 32, CFG.IMG_SIZE[1] // 16, (1,))[0]
            new[:, s:s + w] *= 0
            count += 1
        return new


class HMSHBASpecDatasetADD(HMSHBASpecDataset):
    def __init__(
            self,
            data,
            specs,
            eeg_specs,
            transform: A.Compose,
            phase: str
    ):
        super().__init__()
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5

    def make_Xy(self, new_ind):
        X = np.zeros((512, 512, 4), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        img = np.ones((512, 512), dtype="float32")

        row = self.data.iloc[new_ind]
        r = int((row['min'] + row['max']) // 4)

        # for k in range(1):
        # extract transform spectrogram
        # img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T  # (100, 300)
        img = self.specs[row.spec_id][:, :].T  # (256, 400 or ???)
        # print(row.spec_id, img.shape)
        ch = img.shape[1] // 2
        if ch >= 256:
            img = self.specs[row.spec_id][ch - 256:ch + 256, :].T  # (256, 512)
        else:
            img = self.specs[row.spec_id][:, :].T  # (256, ???)

        # print(row.spec_id, img.shape)
        h, w = img.shape[:2]

        # if w!=256:
        #    print(row.spec_id, img.shape)

        # log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        # crop to 256 time steps
        X[(512 - h) // 2:(512 + h) // 2, (512 - w) // 2:(512 + w) // 2, 0] = img[:, :] / 2.0

        # EEG spectrograms
        img = self.eeg_specs[row.eeg_id]  # (256, 512, 5)

        X[0:256, :, 1] = img[:, :, 0]  # (512, 512, 4)
        X[256:512, :, 1] = img[:, :, 1]  # (512, 512, 4)
        X[0:256, :, 2] = img[:, :, 2]  # (512, 512, 4)
        X[256:512, :, 2] = img[:, :, 3]  # (512, 512, 4)
        X[0:256, :, 3] = img[:, :, 4]  # (512, 512, 4)  fz-cz-pz information

        if self.phase == 'train':
            X = self.spec_mask(X)

            if torch.rand(1) > self.aug_prob:
                X = self.shift_img(X)

        X = self._apply_transform(X)
        X_ = [X[i:i + 1, :, :] for i in range(3)]  # x: [3,512,512]
        X_ += [X[3:4, :256, :]]
        X = torch.cat(X_, dim=1)  # (1, 1536 + 256, 512)

        y[:] = row[CLASSES]

        # if self.phase =='train':
        #      y[:] = self.smooth_labels(row[CLASSES])

        return X, y


class HMSHBASpecDatasetWithoutAug(HMSHBASpecDataset):
    def __init__(
            self,
            data,
            specs,
            eeg_specs,
            transform: A.Compose,
            phase: str
    ):
        # super().__init__()
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5

    def __getitem__(self, index: int):
        X1, y1 = self.make_Xy(index)

        X = torch.cat([X1, X1, X1], dim=0)  # (3, 1536, 512)
        y = (y1 + y1 + y1) / 3

        return {"data": X, "target": y}


class HMSHBASpecDatasetPretrain(HMSHBASpecDataset):
    def __init__(
            self,
            data,
            eeg_specs,
            transform: A.Compose,
            phase: str
    ):
        # super().__init__()
        self.data = data
        self.eeg_specs = eeg_specs
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5

    def make_Xy(self, new_ind):
        X = np.zeros((256, 256, 2), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[new_ind]

        # EEG spectrograms
        img = np.load(f'./EEG_Spectrograms_sparcnet/{row.key}.npy')  # 128,256,4

        X[0:128, :, 0] = img[:, :, 0]  # 256, 256, 2
        X[128:256, :, 0] = img[:, :, 1]  # 256, 256, 2
        X[0:128, :, 1] = img[:, :, 2]  # 256, 256, 2
        X[128:256, :, 1] = img[:, :, 3]  # 256, 256, 2

        if self.phase == 'train':
            X = self.spec_mask(X)

            if torch.rand(1) > self.aug_prob:
                X = self.shift_img(X)

        X = self._apply_transform(X)

        X = [X[i:i + 1, :, :] for i in range(2)]
        X = torch.cat(X, dim=1)  # (1, 512, 256)

        y[:] = row[CLASSES]

        return X, y


class HMSHBASpecDatasetPretrainCWT(HMSHBASpecDataset):
    def __init__(
            self,
            data,
            transform: A.Compose,
            phase: str,
            cfg
    ):
        # super().__init__()
        self.data = data
        self.transform = transform
        self.phase = phase
        self.aug_prob = 0.5
        self.cfg = cfg

    def make_Xy(self, new_ind):
        X = np.zeros((396, 400, 2), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[new_ind]

        # EEG spectrograms
        # print(f'./{self.cfg.pretain_egg_path}/{row.key}.npy')
        img = np.load(f'./{self.cfg.pretain_egg_path}/{row.key}.npy')  # 256,512,4

        X[:198, :, 0] = img[:, :, 0]  # 396, 400, 2
        X[198:, :, 0] = img[:, :, 1]  # 396, 400, 2
        X[:198, :, 1] = img[:, :, 2]  # 396, 400, 2
        X[198:, :, 1] = img[:, :, 3]  # 396, 400, 2

        if self.phase == 'train':
            X = self.spec_mask(X)

            if torch.rand(1) > self.aug_prob:
                X = self.shift_img(X)
        X = self._apply_transform(X)
        X = [X[i:i + 1, :, :] for i in range(2)]
        X = torch.cat(X, dim=1)  # (1, 1024, 512)
        # print(X.sum())
        y[:] = row[CLASSES]

        return X, y

    def __getitem__(self, index: int):

        X1, y1 = self.make_Xy(index)
        if torch.rand(1) > 0.5 and self.phase == 'train':
            index2 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X2, y2 = self.make_Xy(index2)

        else:
            X2, y2 = X1, y1

        if torch.rand(1) > 0.5 and self.phase == 'train':
            index3 = torch.randint(0, self.__len__(), (1,)).numpy()[0]
            X3, y3 = self.make_Xy(index3)

        else:
            X3, y3 = X1, y1
        X = torch.cat([X1, X2, X3], dim=0)  # (3, 1536, 512)
        y = (y1 + y2 + y3) / 3

        return {"data": X, "target": y}


if __name__ == '__main__':
    print(1)
