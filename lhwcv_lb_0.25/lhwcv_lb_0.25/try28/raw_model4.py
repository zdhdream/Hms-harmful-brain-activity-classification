# -*- coding: utf-8 -*-
#import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import math, torch# torchaudio
import random

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    densenet121, inception_v3, resnet18, resnet34, resnet50

from my_model import MyModel2
from raw_model2 import _WaveBlock,_DilatedInception
from cwt import CWT


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8,
                 norm_cls=nn.BatchNorm1d):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = norm_cls(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(norm_cls(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = norm_cls(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

        self.need_permute = True if norm_cls == nn.LayerNorm else False

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        if self.need_permute:
            # print('permute...')
            out = out.permute(0, 2, 1)
        out = self.bn1(out)
        if self.need_permute:
            out = out.permute(0, 2, 1)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            if self.need_permute:
                sp = sp.permute(0, 2, 1)
            sp = self.bns[i](sp)
            if self.need_permute:
                sp = sp.permute(0, 2, 1)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        if self.need_permute:
            out = out.permute(0, 2, 1)
        out = self.bn3(out)
        if self.need_permute:
            out = out.permute(0, 2, 1)
        out = self.se(out)
        out += residual
        return out


import numpy as np


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.w = nn.Parameter(torch.from_numpy(coef * np.ones((1, 1, 1))).float(), requires_grad=True)

    def forward(self, input: torch.tensor) -> torch.tensor:
        # stds = input.std(dim=-1, keepdim=True)
        # input = input / (stds + 1e-3)

        b, c, h = input.shape
        #w = self.w.repeat(b, 12, 1)
        return input * self.w
        # return  self.coef * input


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        # x = self.mask_along_axis(x, dim=1)
        return x


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

def mixup_data_v2(images, images2,  targets, alpha=1.0, use_cuda=True):
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
    mixed_images2 = lam * images2 + (1 - lam) * images2[index, :]

    targets_a, targets_b = targets, targets[index]
    mixed_targets = lam * targets_a + (1 - lam) * targets_b

    return mixed_images, mixed_images2, mixed_targets


class Spec2D(nn.Module):
    def __init__(self, in_c=1, out_c=1, expansion=8, t_dilation=1):
        super(Spec2D, self).__init__()
        kernels_t = [1, 3, 5, 7, 9, 11]

        kernel_f = 3
        self.convs = nn.ModuleList()
        for k in kernels_t:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=expansion,
                    kernel_size=(kernel_f, k),
                    padding="same",
                    dilation=(1, t_dilation)), )
        cs = len(kernels_t) * expansion
        if out_c != cs:
            self.proj = nn.Conv2d(cs, out_c, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        """

        :param x: B, C, F, T
        :return:
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)
        if self.proj is not None:
            h = self.proj(h)
        return h


class ModelCWT(nn.Module):

    def __init__(self,
                 backbone='b2',
                 n_fft=1024,
                 win_length=128,
                 num_mels=128,
                 width=300,
                 with_L_R = True
                 ):
        super(ModelCWT, self).__init__()

        sample_rate = 200
        hop_length = 10000 // width
        print('hop_length: ', hop_length)
        print('n_fft: ', n_fft)
        print('win_length: ', win_length)

        # self.torchfbank = torch.nn.Sequential(
        #     PreEmphasis(coef=0.001),
        #     # torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
        #     #                                      n_fft=n_fft,
        #     #                                      win_length=win_length,
        #     #                                      hop_length=hop_length,
        #     #                                      n_mels=num_mels),
        #     CWT(fmin=0, fmax=25, hop_length=hop_length, dj= 0.081)
        # )
        self.pre = PreEmphasis(coef=0.001)
        self.cwt = CWT(fmin=0, fmax=50, trainable=False,
                       hop_length=hop_length, dj=0.081,)
                       #signal_length=10000, signal_channels=12)

        # self.in_c = num_mels * 8
        #
        # freq_mask_width = (0, 8)
        # time_mask_width = (0, 10)
        # self.specaug = FbankAug(freq_mask_width, time_mask_width)  # Spec augmentation

        pretrained = True
        fea_dim = 6

        if backbone == 'mine':
            self.base_model = MyModel2()
        elif backbone == 'b0':
            self.base_model = efficientnet_b0(pretrained=pretrained)
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, fea_dim)
        elif backbone == 'b1':
            self.base_model = efficientnet_b1(pretrained=pretrained)
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, fea_dim)
        elif backbone == 'b2':
            self.base_model = efficientnet_b2(pretrained=pretrained)
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, fea_dim)
        elif backbone == 'b3':
            self.base_model = efficientnet_b3(pretrained=pretrained)
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, fea_dim)

        elif backbone == 'inception_v3':
            self.base_model = inception_v3(pretrained=pretrained)
            self.base_model.fc = nn.Linear(2048, fea_dim)
        elif backbone == 'densenet121':
            self.base_model = densenet121(pretrained=pretrained)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, fea_dim)
        elif backbone == 'resnet18':
            self.base_model = resnet18(pretrained=pretrained)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, fea_dim)
        elif backbone == 'resnet34':
            self.base_model = resnet34(pretrained=pretrained)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, fea_dim)
        elif backbone == 'resnet50':
            self.base_model = resnet50(pretrained=pretrained)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, fea_dim)
        else:
            import timm
            # hgnet_tiny, hgnet_small, hgnetv2_b0, hgnetv2_b1, hgnetv2_b2
            # tiny_vit_5m_224, tiny_vit_11m_224
            print('create: ', backbone)
            self.base_model = timm.create_model(backbone, pretrained=True)
            self.base_model.reset_classifier(fea_dim)

        self.with_mixup = True
        self.spec = Spec2D(1, 1, 8, 1)

    def forward(self, x, y=None):
        x = x.permute(0, 2, 1)
        with torch.no_grad():
        #if True:
            x = self.cwt(self.pre(x))
            x = x + 1e-6
            x = x.log()
        #print(' x shape: ', x.shape)

        if self.training and self.with_mixup:
            if random.random() < 0.5:
                x, y = mixup_data(x, y)
        b, c, f, t = x.shape

        xs = []
        for i in range(c):
            xs.append(self.spec(x[:, i:i + 1, :, :]))
        x = torch.cat(xs, dim=1)

        x = x.reshape(b, c * f, t)
        x = x.unsqueeze(1)
        x = torch.cat((x, x, x), dim=1)
        x = self.base_model(x)


        if y is not None:
            return x, y
        return x


if __name__ == '__main__':
    model = ModelCWT().cuda().eval()
    x = torch.randn(2, 10000, 12).cuda()
    pred = model(x)
    print('pred shape: ', pred.shape)
