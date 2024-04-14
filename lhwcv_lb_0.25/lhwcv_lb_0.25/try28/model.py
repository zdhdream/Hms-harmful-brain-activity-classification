# -*- coding: utf-8 -*-
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import math, torch, torchaudio
import random

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    densenet121, inception_v3, resnet18, resnet34, resnet50

from my_model import MyModel2
from raw_model2 import _WaveBlock,_DilatedInception


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


class MyModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        pretrained = True
        fea_dim = 6  # 64
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

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            if random.random() < 0.0:
                x, targets = mixup_data(x, targets)
                # x, targets = hbac_cutmix(x, targets,
                #                          margin=0,
                #                          min_size=100,  # Min Cutout Size
                #                          max_size=300,  # Max Cutout Size
                #                          time_size=512)

        out = self.base_model(x)
        if self.training and targets is not None:
            return out, targets
        return out
