import gc
import logging
import math
import os
import random
import sys
import json
import pickle
import warnings
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.signal import butter, lfilter
from sklearn.model_selection import GroupKFold
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

import wandb
from kaggle_kl_div import score


class _Logger:
    """Customized logger.

    Args:
        logging_level: lowest-severity log message the logger handles
        logging_file: file stream for logging
            *Note: If `logging_file` isn't specified, message is only
                logged to system standard output.
    """

    _logger: logging.Logger = None

    def __init__(
            self,
            logging_level: str = "INFO",
            logging_file: Optional[Path] = None,
    ):
        self.logging_level = logging_level
        self.logging_file = logging_file

        self._build_logger()

    def get_logger(self) -> logging.Logger:
        """Return customized logger."""
        return self._logger

    def _build_logger(self) -> None:
        """Build logger."""
        self._logger = logging.getLogger()
        self._logger.setLevel(self._get_level())
        self._add_handler()

    def _get_level(self) -> int:
        """Return lowest severity of the events the logger handles.

        Returns:
            level: severity of the events
        """
        level = 0

        if self.logging_level == "DEBUG":
            level = logging.DEBUG
        elif self.logging_level == "INFO":
            level = logging.INFO
        elif self.logging_level == "WARNING":
            level = logging.WARNING
        elif self.logging_level == "ERROR":
            level = logging.ERROR
        elif self.logging_level == "CRITICAL":
            level = logging.CRITICAL

        return level

    def _add_handler(self) -> None:
        """Add stream and file (optional) handlers to logger."""
        s_handler = logging.StreamHandler(sys.stdout)
        self._logger.addHandler(s_handler)

        if self.logging_file is not None:
            f_handler = logging.FileHandler(self.logging_file, mode="a")
            self._logger.addHandler(f_handler)


def _seed_everything(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Args:
        seed: manually specified seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


DATA_PATH = Path("data")


class CFG:
    train_models = True
    seed = 42

    exp_id = datetime.now().strftime("%m%d-%H-%M-%S")
    exp_dump_path = Path("EEGNet/") / exp_id
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # == Data ==
    gen_eegs = False
    # Chris' 8 channels
    feats = [
        "Fp1", "T3", "C3", "O1",
        "Fp2", "C4", "T4", "O2"
    ]
    cast_eegs = True
    dataset = {
        "eeg": {
            "n_feats": 8,
            "apply_chris_magic_ch8": True,
            "normalize": True,
            "apply_butter_lowpass_filter": True,  # 是否应用低通滤波器
            "apply_mu_law_encoding": False,
            "downsample": 5  # 10_000 // downsample
        }
    }

    # == Trainer ==
    trainer = {
        "epochs": 10,
        "lr": 1e-3,
        "dataloader": {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 2
        },
        "use_amp": True,
        "grad_accum_steps": 1,
        "model_ckpt": {
            "ckpt_metric": "kldiv",
            "ckpt_mode": "min",
            "best_ckpt_mid": "last"
        },
        "es": {"patience": 0},
        "step_per_batch": True,
        "one_batch_only": False
    }

    # == Debug ==
    one_fold_only = False


N_CLASSES = 6
TGT_VOTE_COLS = [
    "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote",
    "grda_vote", "other_vote"
]
TGT_COL = "target"
EEG_FREQ = 200  # Hz
EEG_WLEN = 50  # sec
EEG_PTS = int(EEG_FREQ * EEG_WLEN)  # 200Hz * 50s = 10000

if not CFG.exp_dump_path.exists():
    os.mkdir(CFG.exp_dump_path)

logger = _Logger(logging_file=CFG.exp_dump_path / "train_eval.log").get_logger()
_seed_everything(CFG.seed)


def _get_eeg_window(file: Path) -> np.ndarray:
    """Return cropped EEG window.

    Default setting is to return the middle 50-sec window.

    Args:
        file: EEG file path
        test: if True, there's no need to truncate EEGs

    Returns:
        eeg_win: cropped EEG window
    """
    eeg = pd.read_parquet(file, columns=CFG.feats)
    n_pts = len(eeg)
    offset = (n_pts - EEG_PTS) // 2
    eeg = eeg.iloc[offset:offset + EEG_PTS]

    eeg_win = np.zeros((EEG_PTS, len(CFG.feats)))
    for j, col in enumerate(CFG.feats):
        if CFG.cast_eegs:
            eeg_raw = eeg[col].values.astype("float32")
        else:
            eeg_raw = eeg[col].values

            # Fill missing values
        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else:
            # All missing
            eeg_raw[:] = 0
        eeg_win[:, j] = eeg_raw

    return eeg_win


class _EEGTransformer(object):
    """Data transformer for raw EEG signals."""

    FEAT2CODE = {f: i for i, f in enumerate(CFG.feats)}

    def __init__(
            self,
            n_feats: int,
            apply_chris_magic_ch8: bool = True,
            normalize: bool = True,
            apply_butter_lowpass_filter: bool = True,
            apply_mu_law_encoding: bool = False,
            downsample: Optional[int] = None,
    ) -> None:
        self.n_feats = n_feats  # 8 features
        self.apply_chris_magic_ch8 = apply_chris_magic_ch8
        self.normalize = normalize
        self.apply_butter_lowpass_filter = apply_butter_lowpass_filter
        self.apply_mu_law_encoding = apply_mu_law_encoding
        self.downsample = downsample

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation on raw EEG signals.

        Args:
            x: raw EEG signals, with shape (L, C)

        Return:
            x_: transformed EEG signals
        """
        x_ = x.copy()
        if self.apply_chris_magic_ch8:
            x_ = self._apply_chris_magic_ch8(x_)

        if self.normalize:
            x_ = np.clip(x_, -1024, 1024)
            x_ = np.nan_to_num(x_, nan=0) / 32.0

        if self.apply_butter_lowpass_filter:
            x_ = self._butter_lowpass_filter(x_)

        if self.apply_mu_law_encoding:
            x_ = self._quantize_data(x_, 1)

        if self.downsample is not None:
            x_ = x_[::self.downsample, :]  # (10_000//downsample, 8)

        return x_

    def _apply_chris_magic_ch8(self, x: np.ndarray) -> np.ndarray:
        """Generate features based on Chris' magic formula."""
        x_tmp = np.zeros((EEG_PTS, self.n_feats), dtype="float32")  # (10_000, 8)

        # Generate features
        x_tmp[:, 0] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["T3"]]  # Fp1 - T3
        x_tmp[:, 1] = x[:, self.FEAT2CODE["T3"]] - x[:, self.FEAT2CODE["O1"]]  # T3 -O1

        x_tmp[:, 2] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["C3"]]  # Fp1 - C3
        x_tmp[:, 3] = x[:, self.FEAT2CODE["C3"]] - x[:, self.FEAT2CODE["O1"]]  # C3 - O1

        x_tmp[:, 4] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["C4"]]  # Fp2 - C4
        x_tmp[:, 5] = x[:, self.FEAT2CODE["C4"]] - x[:, self.FEAT2CODE["O2"]]  # C4 - O2

        x_tmp[:, 6] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["T4"]]  # Fp2 - T4
        x_tmp[:, 7] = x[:, self.FEAT2CODE["T4"]] - x[:, self.FEAT2CODE["O2"]]  # T4 - O2

        return x_tmp

    def _butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data, axis=0)

        return filtered_data

    def _quantize_data(self, data, classes):
        mu_x = self._mu_law_encoding(data, classes)

        return mu_x

    def _mu_law_encoding(self, data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)

        return mu_x


class EEGDataset(Dataset):
    """Dataset for pure raw EEG signals.

    Args:
        data: processed data
        split: data split

    Attributes:
        _n_samples: number of samples
        _infer: if True, the dataset is constructed for inference
            *Note: Ground truth is not provided.
    """

    def __init__(
            self,
            data: Dict[str, Any],
            split: str,
            **dataset_cfg: Any,
    ) -> None:
        self.metadata = data["meta"]  # data
        self.all_eegs = data["eeg"]  # brain-eeg
        self.dataset_cfg = dataset_cfg  # the setting in dataset

        # Raw EEG data transformer
        self.eeg_params = dataset_cfg["eeg"]  # the parameters of dataset
        self.eeg_trafo = _EEGTransformer(**self.eeg_params)

        self._set_n_samples()
        self._infer = True if split == "test" else False

        self._stream_X = True if self.all_eegs is None else False
        self._X, self._y = self._transform()

    def _set_n_samples(self) -> None:
        assert len(self.metadata) == self.metadata["eeg_id"].nunique()
        self._n_samples = len(self.metadata)

    def _transform(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Transform feature and target matrices."""
        if self.eeg_params["downsample"] is not None:
            eeg_len = int(EEG_PTS / self.eeg_params["downsample"])
        else:
            eeg_len = int(EEG_PTS)
        if not self._stream_X:
            X = np.zeros((self._n_samples, eeg_len, self.eeg_params["n_feats"]),
                         dtype="float32")  # (num_unique_ids, 10_000/downsample, 8)
        else:
            X = None
        y = np.zeros((self._n_samples, N_CLASSES), dtype="float32") if not self._infer else None

        for i, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            # Process raw EEG signals
            if not self._stream_X:
                # Retrieve raw EEG signals
                eeg = self.all_eegs[row["eeg_id"]]

                # Apply EEG transformer
                x = self.eeg_trafo.transform(eeg)  # (10_000/downsample, 8)

                X[i] = x

            if not self._infer:
                y[i] = row[TGT_VOTE_COLS]

        return X, y

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self._X is None:
            # Load data here...
            #             x = np.load(...)
            #             x = self.eeg_trafo.transform(x)
            pass
        else:
            x = self._X[idx, ...]
        data_sample = {"x": torch.tensor(x, dtype=torch.float32)}
        if not self._infer:
            data_sample["y"] = torch.tensor(self._y[idx, :], dtype=torch.float32)

        return data_sample


class _WaveBlock(nn.Module):
    """WaveNet block.

    Args:
        kernel_size: kernel size, pass a list of kernel sizes for
            inception
    """

    def __init__(
            self,
            n_layers: int,
            in_dim: int,  # in_channels
            h_dim: int,  # out_channels
            kernel_size: Union[int, List[int]],
            conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers  # 12
        self.dilation_rates = [2 ** l for l in range(n_layers)]

        self.in_conv = nn.Conv2d(in_dim, h_dim, kernel_size=(1, 1))
        self.gated_tcns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            c_in, c_out = h_dim, h_dim
            self.gated_tcns.append(
                _GatedTCN(
                    in_dim=c_in,
                    h_dim=c_out,
                    kernel_size=kernel_size,
                    dilation_factor=self.dilation_rates[layer],
                    conv_module=conv_module,
                )
            )
            self.skip_convs.append(nn.Conv2d(h_dim, h_dim, kernel_size=(1, 1)))

        # Initialize parameters
        nn.init.xavier_uniform_(self.in_conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.in_conv.bias)
        for i in range(len(self.skip_convs)):
            nn.init.xavier_uniform_(self.skip_convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.skip_convs[i].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, C, N, L), where C denotes in_dim
            x_skip: (B, C', N, L), where C' denotes h_dim
        """
        # Input convolution
        x = self.in_conv(x)

        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated_tcns[layer](x)
            x = self.skip_convs[layer](x)

            # Skip-connection
            x_skip = x_skip + x

        return x_skip


class _GatedTCN(nn.Module):
    """Gated temporal convolution layer.

    Parameters:
        conv_module: customized convolution module
    """

    def __init__(
            self,
            in_dim: int,
            h_dim: int,
            kernel_size: Union[int, List[int]],
            dilation_factor: int,  # 膨胀因子
            dropout: Optional[float] = None,
            conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # Model blocks
        if conv_module is None:
            self.filt = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
            self.gate = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
        else:
            self.filt = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )
            self.gate = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            h: (B, h_dim, N, L')
        """
        x_filt = F.tanh(self.filt(x))  # 滤波
        x_gate = F.sigmoid(self.gate(x))  # 门控
        h = x_filt * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class _DilatedInception(nn.Module):
    """Dilated inception layer.

    Note that `out_channels` will be split across #kernels.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: List[int],
            dilation: int
    ) -> None:
        super().__init__()

        # Network parameters
        n_kernels = len(kernel_size)
        assert out_channels % n_kernels == 0, "`out_channels` must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        # Model blocks
        self.convs = nn.ModuleList()
        for k in kernel_size:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=(1, k),
                    padding="same",
                    dilation=dilation),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where C = in_channels
            h: (B, C', N, L'), where C' = out_channels
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)

        return h


class DilatedInceptionWaveNet(nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(self, ) -> None:
        super().__init__()

        kernel_size = [2, 3, 6, 7]

        # Model blocks
        self.wave_module = nn.Sequential(
            _WaveBlock(12, 1, 16, kernel_size, _DilatedInception),
            _WaveBlock(8, 16, 32, kernel_size, _DilatedInception),
            _WaveBlock(4, 32, 64, kernel_size, _DilatedInception),
            _WaveBlock(1, 64, 64, kernel_size, _DilatedInception),
        )
        self.output = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES)
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, L, C)
        """
        x = inputs["x"]  # (bs,10_000//downsample,8)
        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant  (bs, 8, 1, 2000)

        x_ll_1 = self.wave_module(x[:, 0:1, :])  # LT (bs,8,1,2000) -> (bs,64,1,2000)
        x_ll_2 = self.wave_module(x[:, 1:2, :])  # LP (bs,8,1,2000) -> (bs,64,1,2000)
        x_ll = (F.adaptive_avg_pool2d(x_ll_1, (1, 1)) + F.adaptive_avg_pool2d(x_ll_2, (1, 1))) / 2 # (32, 64, 1, 1)

        x_rl_1 = self.wave_module(x[:, 2:3, :])  # RP
        x_rl_2 = self.wave_module(x[:, 3:4, :])  # RT
        x_rl = (F.adaptive_avg_pool2d(x_rl_1, (1, 1)) + F.adaptive_avg_pool2d(x_rl_2, (1, 1))) / 2

        x_lp_1 = self.wave_module(x[:, 4:5, :])
        x_lp_2 = self.wave_module(x[:, 5:6, :])
        x_lp = (F.adaptive_avg_pool2d(x_lp_1, (1, 1)) + F.adaptive_avg_pool2d(x_lp_2, (1, 1))) / 2

        x_rp_1 = self.wave_module(x[:, 6:7, :])
        x_rp_2 = self.wave_module(x[:, 7:8, :])
        x_rp = (F.adaptive_avg_pool2d(x_rp_1, (1, 1)) + F.adaptive_avg_pool2d(x_rp_2, (1, 1))) / 2

        x = torch.cat([x_ll, x_rl, x_lp, x_rp], axis=1).reshape(bs, -1) # (32, 256)
        output = self.output(x)

        return output


class KLDivWithLogitsLoss(nn.KLDivLoss):
    """Kullback-Leibler divergence loss with logits as input."""

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = F.log_softmax(y_pred, dim=1)
        kldiv_loss = super().forward(y_pred, y_true)

        return kldiv_loss


class Evaluator(object):
    """Custom evaluator.

    Args:
        metric_names: evaluation metrics
    """

    eval_metrics: Dict[str, Callable[..., float]] = {}
    EPS: float = 1e-6

    def __init__(self, metric_names: List[str]) -> None:
        self.metric_names = metric_names

        self._build()

    def evaluate(
            self,
            y_true: Tensor,
            y_pred: Tensor,
            scaler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Args:
            y_true: ground truth
            y_pred: prediction
            scaler: scaling object

        Returns:
            eval_result: evaluation performance report
        """
        if scaler is not None:
            # Do inverse transformation to rescale y values
            y_pred, y_true = self._rescale_y(y_pred, y_true, scaler)

        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            eval_result[metric_name] = metric(y_pred, y_true).item()

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "kldiv":
                self.eval_metrics[metric_name] = KLDivWithLogitsLoss()
            elif metric_name == "ce":
                self.eval_metrics[metric_name] = nn.CrossEntropyLoss()

    def _rescale_y(self, y_pred: Tensor, y_true: Tensor, scaler: Any) -> Tuple[Tensor, Tensor]:
        """Rescale y to the original scale.

        Args:
            y_pred: prediction
            y_true: ground truth
            scaler: scaling object

        Returns:
            y_pred: rescaled prediction
            y_true: rescaled ground truth
        """
        # Do inverse transform...

        return y_pred, y_true


class _ModelCheckpoint(object):
    """Model checkpooint.

    Args:
        ckpt_path: path to save model checkpoint
        ckpt_metric: quantity to monitor during training process
        ckpt_mode: determine the direction of metric improvement
        best_ckpt_mid: model identifier of the probably best checkpoint
            used to do the final evaluation
    """

    def __init__(self, ckpt_path: Path, ckpt_metric: str, ckpt_mode: str, best_ckpt_mid: str) -> None:
        self.ckpt_path = ckpt_path
        self.ckpt_metric = ckpt_metric
        self.ckpt_mode = ckpt_mode
        self.best_ckpt_mid = best_ckpt_mid

        # Specify checkpoint direction
        self.ckpt_dir = -1 if ckpt_mode == "max" else 1

        # Initialize checkpoint status
        self.best_val_score = 1e18
        self.best_epoch = 0

    def step(
            self, epoch: int, model: nn.Module, val_loss: float, val_result: Dict[str, float], last_epoch: bool = False
    ) -> None:
        """Update checkpoint status for the current epoch.

        Args:
            epoch: current epoch
            model: current model instance
            val_loss: validation loss
            val_result: evaluation result on validation set
            last_epoch: if True, current epoch is the last one
        """
        val_score = val_loss if self.ckpt_metric is None else val_result[self.ckpt_metric]
        val_score = val_score * self.ckpt_dir
        if val_score < self.best_val_score:  # type: ignore
            logging.info(f"Validation performance improves at epoch {epoch}!!")
            self.best_val_score = val_score
            self.best_epoch = epoch

            # Save model checkpoint
            mid = "loss" if self.ckpt_metric is None else self.ckpt_metric
            self._save_ckpt(model, mid)

        if last_epoch:
            self._save_ckpt(model, "last")

    def save_ckpt(self, model: nn.Module, mid: Optional[str] = None) -> None:
        """Save the checkpoint.

        Args:
            model: current model instance
            mid: model identifer
        """
        self._save_ckpt(model, mid)

    def load_best_ckpt(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Load and return the best model checkpoint for final evaluation.

        Args:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint.
            device: device of the model instance

        Returns:
            best_model: best model checkpoint
        """
        model = self._load_ckpt(model, device, self.best_ckpt_mid)

        return model

    def _save_ckpt(self, model: nn.Module, mid: Optional[str] = None) -> None:
        """Save the model checkpoint.

        Args:
            model: current model instance
            mid: model identifer
        """
        model_file = "model.pth" if mid is None else f"model-{mid}.pth"
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, model_file))

    def _load_ckpt(self, model: nn.Module, device: torch.device, mid: str = "last") -> nn.Module:
        """Load the model checkpoint.

        Args:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint.
            device: device of the model instance
            mid: model identifier

        Returns:
            model: model instance with the loaded weights
        """
        model_file = f"model-{mid}.pth"
        model.load_state_dict(torch.load(os.path.join(self.ckpt_path, model_file), map_location=device))

        return model


class _BaseTrainer:
    """Base class for all customized trainers.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        ckpt_path: path to save model checkpoints
        es: early stopping tracker
        evaluator: task-specific evaluator
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    train_loader: DataLoader  # Tmp. workaround
    eval_loader: DataLoader  # Tmp. workaround

    def __init__(
            self,
            logger: _Logger,
            trainer_cfg: Dict[str, Any],
            model: nn.Module,
            loss_fn: _Loss,
            optimizer: Optimizer,
            lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
            ckpt_path: Path,
            evaluator: Evaluator,
            use_wandb: bool = False,
    ):
        self.logger = logger
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.ckpt_path = ckpt_path
        self.evaluator = evaluator
        self.use_wandb = use_wandb

        self.device = CFG.device
        self.epochs = trainer_cfg["epochs"]
        self.use_amp = trainer_cfg["use_amp"]
        self.grad_accum_steps = trainer_cfg["grad_accum_steps"]
        self.step_per_batch = trainer_cfg["step_per_batch"]

        # Debug options
        self.one_batch_only = trainer_cfg["one_batch_only"]

        # Model checkpoint
        self.model_ckpt = _ModelCheckpoint(ckpt_path, **trainer_cfg["model_ckpt"])

        # Early stopping
        if trainer_cfg["es"]["patience"] != 0:
            self.logger.info("Please disable early stop!")
        #             self.es = EarlyStopping(**trainer_cfg["es"])
        else:
            self.es = None

        self._iter = 0
        self._track_best_model = True  # (Deprecated)

    def train_eval(self, proc_id: int) -> Dict[str, np.ndarray]:
        """Run training and evaluation processes.

        Args:
            proc_id: identifier of the current process
        """
        self.logger.info("Start training and evaluation processes...")
        for epoch in range(self.epochs):
            self.epoch = epoch  # For interior use
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None and not self.step_per_batch:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result (by epoch)
            self._log_proc(epoch, train_loss, val_loss, val_result)

            # Record the best checkpoint
            self.model_ckpt.step(
                epoch, self.model, val_loss, val_result, last_epoch=False if epoch != self.epochs - 1 else True
            )

            # Check early stopping is triggered or not
            if self.es is not None:
                self.es.step(val_loss)
                if self.es.stop:
                    self.logger.info(f"Early stopping is triggered at epoch {epoch}, training process is halted.")
                    break
        if self.use_wandb:
            wandb.log({"best_epoch": self.model_ckpt.best_epoch + 1})  # `epoch` starts from 0

        # Run final evaluation
        final_prf_report, y_preds = self._run_final_eval()
        self._log_best_prf(final_prf_report)

        return y_preds

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
                *Note: If MTL is used, returned object will be dict
                    containing losses of sub-tasks and the total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, return_output: bool = False) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        raise NotImplementedError

    def _log_proc(
            self,
            epoch: int,
            train_loss: Union[float, Dict[str, float]],
            val_loss: Optional[float] = None,
            val_result: Optional[Dict[str, float]] = None,
            proc_id: Optional[str] = None,
    ) -> None:
        """Log message of training process.

        Args:
            epoch: current epoch number
            train_loss: training loss
            val_loss: validation loss
            val_result: evaluation performance report
            proc_id: identifier of the current process
        """
        proc_msg = [f"Epoch{epoch} [{epoch + 1}/{self.epochs}]"]

        # Construct training loss message
        if isinstance(train_loss, float):
            proc_msg.append(f"Training loss {train_loss:.4f}")
        else:
            for loss_k, loss_v in train_loss.items():
                loss_name = loss_k.split("_")[0].capitalize()
                proc_msg.append(f"{loss_name} loss {round(loss_v, 4)}")

        # Construct eval prf message
        if val_loss is not None:
            proc_msg.append(f"Validation loss {val_loss:.4f}")
        if val_result is not None:
            for metric, score in val_result.items():
                proc_msg.append(f"{metric.upper()} {round(score, 4)}")

        proc_msg = " | ".join(proc_msg)
        self.logger.info(proc_msg)

        if self.use_wandb:
            # Process loss dict and log
            log_dict = train_loss if isinstance(train_loss, dict) else {"train_loss": train_loss}
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            if val_result is not None:
                for metric, score in val_result.items():
                    log_dict[metric] = score

            if proc_id is not None:
                log_dict = {f"{k}_{proc_id}": v for k, v in log_dict.items()}

            wandb.log(log_dict)

    def _run_final_eval(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        """Run final evaluation process with designated model checkpoint.

        Returns:
            final_prf_report: performance report of final evaluation
            y_preds: prediction on different datasets
        """
        # Load the best model checkpoint
        self.model = self.model_ckpt.load_best_ckpt(self.model, self.device)

        # Reconstruct dataloaders
        self._disable_shuffle()
        val_loader = self.eval_loader

        final_prf_report, y_preds = {}, {}
        for data_split, dataloader in {
            # "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            _, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[data_split] = eval_result
            y_preds[data_split] = y_pred.numpy()

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Args:
            prf_report: performance report
        """
        self.logger.info(">>>>> Performance Report - Best Ckpt <<<<<")
        self.logger.info(json.dumps(prf_report, indent=4))

        if self.use_wandb:
            wandb.log(prf_report)


class MainTrainer(_BaseTrainer):
    """Main trainer.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    def __init__(
            self,
            logger: _Logger,
            trainer_cfg: Dict[str, Any],
            model: nn.Module,
            loss_fn: _Loss,
            optimizer: Optimizer,
            lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
            ckpt_path: Path,
            evaluator: Evaluator,
            scaler: Any,
            train_loader: DataLoader,
            eval_loader: Optional[DataLoader] = None,
            use_wandb: bool = False,
    ):
        super(MainTrainer, self).__init__(
            logger,
            trainer_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler

        self.loss_name = self.loss_fn.__class__.__name__

        # Mixed precision training
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            if i % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            with autocast(enabled=self.use_amp):
                # Forward pass and derive loss
                output = self.model(inputs)
                loss = self.loss_fn(output, y)
            train_loss_total += loss.item()
            loss = loss / self.grad_accum_steps

            # Backpropagation
            self.grad_scaler.scale(loss).backward()
            if (i + 1) % self.grad_accum_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.step_per_batch:
                    self.lr_skd.step()

            self._iter += 1

            # Free mem.
            del inputs, y, output
            _ = gc.collect()

            if self.one_batch_only:
                break

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
            self,
            return_output: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        eval_loss_total = 0
        y_true, y_pred = [], []

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass
            output = self.model(inputs)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            y_true.append(y.detach().cpu())
            y_pred.append(output.detach().cpu())

            del inputs, y, output
            _ = gc.collect()

        eval_loss_avg = eval_loss_total / len(self.eval_loader)

        # Run evaluation with the specified evaluation metrics
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None


def main():
    train = pd.read_csv("data/train.csv")
    logger.info(f'Train data shape | {train.shape}')
    uniq_eeg_ids = train['eeg_id'].unique()
    n_uniq_eeg_ids = len(uniq_eeg_ids)

    if CFG.gen_eegs:
        logger.info("Generate cropped EEGs...")
        all_eegs = {}
        for i, eeg_id in tqdm(enumerate(uniq_eeg_ids), total=n_uniq_eeg_ids):
            eeg_win = _get_eeg_window(DATA_PATH / "train_eegs" / f"{eeg_id}.parquet")
            all_eegs[eeg_id] = eeg_win
    else:
        logger.info("Load cropped EEGs...")
        all_eegs = np.load("data/brain-eegs/eegs.npy", allow_pickle=True).item()
        assert len(all_eegs) == n_uniq_eeg_ids
    logger.info(f"Demo EEG shape | {list(all_eegs.values())[0].shape}")

    logger.info(f"Process labels...")
    df_tmp = train.groupby("eeg_id")[["patient_id"]].agg("first")
    labels_tmp = train.groupby("eeg_id")[TGT_VOTE_COLS].agg("sum")
    for col in TGT_VOTE_COLS:
        df_tmp[col] = labels_tmp[col].values

    # Normalize target columns
    y_data = df_tmp[TGT_VOTE_COLS].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    df_tmp[TGT_VOTE_COLS] = y_data

    tgt = train.groupby("eeg_id")[["expert_consensus"]].agg("first")
    df_tmp[TGT_COL] = tgt

    train = df_tmp.reset_index()
    logger.info(f"Training DataFrame shape | {train.shape}")

    if CFG.train_models:
        oof = np.zeros((len(train), N_CLASSES))
        prfs = []

        cv = GroupKFold(n_splits=5)
        for fold, (tr_idx, val_idx) in enumerate(cv.split(train, train[TGT_COL], train["patient_id"])):
            logger.info(f"== Train and Eval Process - Fold{fold} ==")

            # Build dataloaders
            data_tr, data_val = train.iloc[tr_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
            train_loader = DataLoader(
                EEGDataset({"meta": data_tr, "eeg": all_eegs}, "train", **CFG.dataset),
                shuffle=CFG.trainer["dataloader"]["shuffle"],
                batch_size=CFG.trainer["dataloader"]["batch_size"],
                num_workers=CFG.trainer["dataloader"]["num_workers"]
            )
            val_loader = DataLoader(
                EEGDataset({"meta": data_val, "eeg": all_eegs}, "valid", **CFG.dataset),
                shuffle=False,
                batch_size=CFG.trainer["dataloader"]["batch_size"],
                num_workers=CFG.trainer["dataloader"]["num_workers"]
            )

            # Build model
            logger.info(f"Build model...")
            model = DilatedInceptionWaveNet()
            model.to(CFG.device)

            # Build criterion
            loss_fn = KLDivWithLogitsLoss()

            # Build solvers
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG.trainer["lr"])
            num_training_steps = (
                    math.ceil(
                        len(train_loader.dataset)
                        / (CFG.trainer["dataloader"]["batch_size"] * CFG.trainer["grad_accum_steps"])
                    )
                    * CFG.trainer["epochs"]
            )
            lr_skd = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                     num_training_steps=num_training_steps)

            # Build evaluator
            evaluator = Evaluator(metric_names=["kldiv"])

            # Build trainer
            trainer: _BaseTrainer = None
            trainer = MainTrainer(
                logger=logger,
                trainer_cfg=CFG.trainer,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_skd=lr_skd,
                ckpt_path=CFG.exp_dump_path,
                evaluator=evaluator,
                scaler=None,
                train_loader=train_loader,
                eval_loader=val_loader,
                use_wandb=False
            )

            # Run main training and evaluation for one fold
            y_preds = trainer.train_eval(fold)
            oof[val_idx, :] = y_preds["val"]

            # Dump output objects
            for model_path in CFG.exp_dump_path.glob("*.pth"):
                if "seed" in str(model_path) or "fold" in str(model_path):
                    continue

                # Rename model file
                model_file_name_dst = f"{model_path.stem}_fold{fold}.pth"
                model_path_dst = CFG.exp_dump_path / model_file_name_dst
                model_path.rename(model_path_dst)

            # Free mem.
            del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
            _ = gc.collect()

            if CFG.one_fold_only:
                logger.info("Cross-validatoin stops at first fold!!!")
                break

        np.save(CFG.exp_dump_pat / "oof.npy", oof)
    else:
        oof = np.load("hms-eeg-oof-demo/oof_seed0.npy")

    # OOf prediction
    y_pred = pd.DataFrame(F.softmax(torch.tensor(oof), dim=1))
    y_pred["id"] = np.arange(len(oof))

    # Ground truth
    y_true = pd.DataFrame(train[TGT_VOTE_COLS].values)
    y_true["id"] = np.arange(len(y_true))

    cv_score = score(solution=y_true, submission=y_pred, row_id_column_name="id")
    logger.info(">>>>> Performance Report <<<<<")
    logger.info(f"-> KL Divergence: {cv_score:.4f}")


if __name__ == "__main__":
    main()
