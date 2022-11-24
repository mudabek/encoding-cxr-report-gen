#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import torch
from torch import relu
from torch.nn import Dropout, Linear, Sequential
from torch.nn.functional import adaptive_avg_pool2d, cross_entropy
from torch.nn.utils import clip_grad_norm_
from torchvision import models
from clinicgen.utils import data_cuda

from custom_models import *


class ImageClassification(torch.nn.Module):
    def __init__(self, model, num_labels, num_classes, multi_image=1, dropout=0.0, pretrained=True):
        super(ImageClassification, self).__init__()
        self.image_feats, self.image_dim = self.image_features(model, False, pretrained)
        for i in range(num_labels):
            setattr(self, 'linear{0}'.format(i), Linear(self.image_dim, num_classes))
        self.num_labels = num_labels
        self.multi_image = multi_image
        self.dropout = Dropout(p=dropout)

    @classmethod
    def fix_layers(cls, model):
        for param in model.parameters():
            param.requires_grad = False

    @classmethod
    def image_features(cls, name, fixed_weight=False, pretrained=True, pretrained_model=None, device='gpu'):
        m = CustomEncoder()
        feature_dim = m.feature_dim
        return m, feature_dim

    def deflatten_image(self, x):
        if self.multi_image > 1:
            x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, x.shape[1])
            x, _ = torch.max(x, dim=1)
        return x

    def flatten_image(self, x):
        if self.multi_image > 1:
            return x.flatten(start_dim=0, end_dim=1)
        else:
            return x

    def forward(self, x):
        x = self.flatten_image(x)
        x = self.image_feats(x)
        x = relu(x)
        x = adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.deflatten_image(x)
        xs = []
        for i in range(self.num_labels):
            xi = self.dropout(x)
            xi = getattr(self, 'linear{0}'.format(i))(xi).unsqueeze(dim=2)
            xs.append(xi)
        x = torch.cat(xs, dim=2)
        return x

    def train_step(self, inp, targ, optimizer, clip_grad=None, device='gpu'):
        optimizer.zero_grad()
        inp, targ = data_cuda(inp, targ, device=device, non_blocking=False)
        targ = targ.squeeze(dim=-1)
        out = self.forward(inp)
        out = out.squeeze(dim=-1)
        loss = cross_entropy(out, targ, ignore_index=-100, reduction='mean')
        loss.backward()
        loss_val = loss.detach().cpu()
        if clip_grad is not None:
            clip_grad_norm_(self.parameters(), clip_grad)
        optimizer.step()
        return loss_val
