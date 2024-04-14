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
from timm.utils import ModelEmaV3, ModelEmaV2
from tqdm import tqdm
from cfg import CFG
from models import HMSHBACSpecModelSED, HMSHBACSpecModel
from datasets import HMSHBASpecDataset, HMSHBASpecDatasetADD, HMSHBASpecDatasetWithoutAug
from mix_up import Mixup
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

RANDOM_SEED =8620
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)

N_WORKERS = 1 #os.cpu_count()


    
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

		weighted_loss = sample_loss*sample_weight
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
    ], weight_decay=decay) # optim.AdamW MADGRAD


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


def train_one_fold(CFG, fold, train_index, val_index, train_all, spectrograms, all_eegs,output_path):
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)

    train_transform, val_transform = get_transforms(CFG)

    train_dataset = HMSHBASpecDataset(train_all.iloc[train_index],
                                      specs=spectrograms,
                                      eeg_specs=all_eegs,
                                      transform=train_transform, phase='train')

    val_dataset = HMSHBASpecDataset(train_all.iloc[val_index],
                                    specs=spectrograms,
                                    eeg_specs=all_eegs,
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
    model_ema = ModelEmaV2(model, decay=0.99, device=None,)# use_warmup=True, warmup_power=3/4)
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
    loss_func = KLDivLossWithLogits() #WeightedKLDivWithLogitsLoss(class_weight_torch)
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
            #x, t = mixup_fn(x,t)
            optimizer.zero_grad()
            with amp.autocast(use_amp, dtype=torch.bfloat16):
                y_0 = model(x)
                loss_0 = loss_func(y_0, t)
                
#                 y_1 = model(x)
#                 loss_1 = loss_func(y_1, t)
                
#                 loss_r = compute_kl_loss(y_0, y_1)
                
#                 alpha = min(5, np.log(epoch))
                
                loss = loss_0 #0.5 * (loss_0 + loss_1) + alpha * loss_r
                
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
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}sec")

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


def main():
    df = pd.read_csv(f"{DATA}/train.csv")

    train = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
    train = train.rename(columns={"spectrogram_id": "spec_id", "spectrogram_label_offset_seconds": "min"})

    tmp = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds': 'max'})
    train['max'] = tmp

    tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp

    tmp = df.groupby('eeg_id')[CLASSES].agg('sum')
    for t in CLASSES:
        train[t] = tmp[t].values

    y_data = train[CLASSES].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[CLASSES] = y_data

    tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train['target'] = tmp

    tmp = df.groupby('eeg_id')[['label_id']].agg('first')
    train['label_id'] = tmp

    train = train.reset_index()

    spectrograms = np.load(f"specs.npy", allow_pickle=True).item()
    all_eegs = np.load(CFG.chris_data, allow_pickle=True).item()
    #all_eegs_aug = np.load(f"eeg_specs_256x512_aug.npy", allow_pickle=True).item()
    score_list = []
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    #gkf = GroupKFold(n_splits=N_FOLDS)
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
