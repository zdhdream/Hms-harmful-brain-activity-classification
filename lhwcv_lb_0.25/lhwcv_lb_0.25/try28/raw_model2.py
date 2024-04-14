# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union, Type, List

import torchaudio
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, efficientnet_b1
import numpy as np

from try28.raw_model3 import Spec2D
from try28.raw_model3 import PreEmphasis


def mixup_data(images, targets, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index, :]
    targets_a, targets_b = targets, targets[index]
    mixed_targets = lam * targets_a + (1 - lam) * targets_b

    return mixed_images, mixed_targets


class _WaveBlock(nn.Module):
    """WaveNet block.

    Args:
        kernel_size: kernel size, pass a list of kernel sizes for
            inception
    """

    def __init__(
            self,
            n_layers: int,
            in_dim: int,
            h_dim: int,
            kernel_size: Union[int, List[int]],
            conv_module: Optional[Type[nn.Module]] = None,
            stride=1,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
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
        self.stride = stride
        if stride != 1:
            ksize = kernel_size[-1]
            self.out_conv = nn.Conv2d(h_dim, h_dim, kernel_size=(1, ksize), padding=(0, ksize // 2), stride=stride)

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
        if self.stride != 1:
            x_skip = self.out_conv(x_skip)
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
            dilation_factor: int,
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
        x_filt = torch.tanh(self.filt(x))
        x_gate = torch.sigmoid(self.gate(x))
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


def hbac_cutmix(data,
                target,
                cutmix_thr=0.75,  # Threshold to consider samples
                margin=50,  # Either Cut 50 from begining  or end
                min_size=25,  # Min Cutout Size
                max_size=75,  # Max Cutout Size
                time_size=300,
                ):
    # CutMix Data
    cutmix_data = data.clone()

    for label_idx in range(target.size(1)):
        # Indices with a confidence score greater than cutmix_thr for particular target
        indices = torch.nonzero((target[:, label_idx] >= cutmix_thr), as_tuple=False).squeeze()

        # Skip if less than 2 samples with coconfidence score 1.0
        if indices.numel() < 2:
            continue

        # Original Data
        data_orig = data[indices]

        # Shuffle
        shuffled_indices = torch.randperm(len(indices))
        data_shuffled = data_orig[shuffled_indices]

        # CutMix augmentation logic
        start = random.randint(0, margin) if random.choice([True, False]) else random.randint(
            time_size - max_size - margin, time_size - max_size)
        size = random.randint(min_size, max_size)

        # CutMix in Specs
        cutmix_data[indices, :, :, start:start + size] = data_shuffled[..., start:start + size]

    return cutmix_data, target


class DilatedInceptionWaveNet(nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(self, kernel_size=None, stride_list=None) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = [2, 3, 6, 7]
        # kernel_size = [3, 5, 7, 9]
        print('kernel_size: ', kernel_size)
        fea_dim = 64
        if stride_list is None:
            stride_list = [2, 2, 2]
        # Model blocks
        self.wave_module = nn.Sequential(
            _WaveBlock(2, 1, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[0]),
            _WaveBlock(2, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[1]),
            _WaveBlock(2, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[2]),
            # nn.Conv2d(in_channels=fea_dim // 4, out_channels=1, kernel_size=(1, 1)),

            _WaveBlock(12, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=2),
            _WaveBlock(8, fea_dim // 4, fea_dim // 2, kernel_size, _DilatedInception, stride=1),
            _WaveBlock(8, fea_dim // 2, fea_dim, kernel_size, _DilatedInception, stride=1),
            # _WaveBlock(2, fea_dim, fea_dim, kernel_size, _DilatedInception),
        )
        self.wave_module2 = _WaveBlock(2, 12 * fea_dim, fea_dim, kernel_size, _DilatedInception)
        self.output = nn.Sequential(
            nn.Linear(2*fea_dim, fea_dim),
            nn.ReLU(),
            nn.Linear(fea_dim, 6)
        )
        self.with_mixup = True
        # self.fuse = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=(1, 1), bias=False)
        hop_length = 10000 // 300
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(coef=0.001),
            torchaudio.transforms.MelSpectrogram(sample_rate=200,
                                                 n_fft=1024,
                                                 win_length=128,
                                                 hop_length=hop_length,
                                                 n_mels=96), )

        pretrained = True

        self.base_model = efficientnet_b0(pretrained=pretrained)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features,
                                                  fea_dim)

        self.spec = Spec2D(1, 1, 16, 1)

    def forward_mel(self, x):
        x = x.permute(0, 2, 1)
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()

        b, c, f, t = x.shape
        # x = self.base_model(x)
        xs = []
        for i in range(c):
            xs.append(self.spec(x[:, i:i + 1, :, :]))
        x = torch.cat(xs, dim=1)

        x = x.reshape(b, c * f, t)
        x = x.unsqueeze(1)
        x = torch.cat((x, x, x), dim=1)
        x = self.base_model(x)

        return x


    def forward(self, x, targets=None) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, L, C)
        """
        x2 = self.forward_mel(x)

        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant

        xs = []
        for i in range(12):
            xs.append(self.wave_module(x[:, i:i + 1, :]))
        x = torch.cat(xs, dim=2)
        bs, fdim, c, t = x.shape
        x = x.reshape(bs, fdim * c, 1, t)

        x = self.wave_module2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(bs, -1)
        x = torch.cat((x2, x),dim=1)
        output = self.output(x)
        if self.training and targets is not None:
            return output, targets
        return output


class DilatedInceptionWaveNetV2(nn.Module):
    def __init__(self, kernel_size=None, stride_list=None) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = [1, 3, 5, 7, 9, 11]
        # kernel_size = [3, 5, 7, 9]
        print('kernel_size: ', kernel_size)
        fea_dim = 72
        if stride_list is None:
            stride_list = [2, 2, 2, 2, 2]
        # Model blocks
        self.wave_module0 = nn.Sequential(
            _WaveBlock(2, 1, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[0]),
            _WaveBlock(2, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[1]),
            _WaveBlock(2, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[2]),
            _WaveBlock(4, fea_dim // 4, fea_dim // 4, kernel_size, _DilatedInception, stride=stride_list[3]),
            _WaveBlock(4, fea_dim // 4, fea_dim // 2, kernel_size, _DilatedInception, stride=stride_list[4])
        )

        self.wave_module1 = _WaveBlock(6, fea_dim // 2, fea_dim // 2, kernel_size, _DilatedInception, stride=1)
        self.wave_module2 = _WaveBlock(8, fea_dim // 2, fea_dim, kernel_size, _DilatedInception, stride=1)

        self.base_model = efficientnet_b1(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 6)

        self.with_mixup = True

    def forward(self, x, y=None):
        """Forward pass.

        Shape:
            x: (B, L, C)
        """

        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant

        xs = []
        for i in range(12):
            xi = self.wave_module0(x[:, i:i + 1, :])
            xi_1 = self.wave_module1(xi)
            xi_2 = self.wave_module2(xi_1)
            xi = torch.cat((xi, xi_1, xi_2), dim=1)
            xs.append(xi)

        x = torch.cat(xs, dim=2).permute(0, 2, 1, 3)
        b, c, f, t = x.shape
        if self.training and self.with_mixup:
            if random.random() < 0.5:
                x, y = mixup_data(x, y)

        x = x.reshape(b, c * f, t)
        x = x.unsqueeze(1)
        #print('x shape: ', x.shape)
        x = torch.cat((x, x, x), dim=1)
        x = self.base_model(x)
        # print('x shape: ',x.shape)

        if self.training and y is not None:
            return x, y
        return x


if __name__ == '__main__':
    L = 2000 * 5
    x = torch.zeros(1, L, 12)
    model = DilatedInceptionWaveNetV2().eval()
    # model.load_state_dict(torch.load('/home/hw/m2_disk/kaggle/working/try8_aug_seed_2024/model-last.pth'))
    inp = x
    y = model(inp)
