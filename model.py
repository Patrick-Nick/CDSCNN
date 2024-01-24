
"""
@FileName: model.py
@Author: Chenghong Xiao
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ComplexConv1d, ComplexBatchNorm, CReLU, ComplexAvgPool2d, ComplexLinear, ComplexMaxPool1d
from CDSC import CDSC1d


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = ComplexConv1d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = ComplexBatchNorm(out_filters)
        else:
            self.skip = None

        self.relu = CReLU()
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(CDSC1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(ComplexBatchNorm(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(CDSC1d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(ComplexBatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(CDSC1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(ComplexBatchNorm(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = CReLU()

        if strides != 1:
            rep.append(ComplexMaxPool1d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        x = x.cuda()

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        skip = skip.cuda()
        x += skip
        return x


class CDSCNN(nn.Module):
    """
    Complex-Valued Depthwise Separable Convolutional Neural Network (CDSCNN)
    """

    def __init__(self, num_classes):
        """ Initialize a CDSCNN
        Args:
            num_classes (int): the number of classes
        """
        super(CDSCNN, self).__init__()

        self.num_classes = num_classes

        self.conv1 = ComplexConv1d(2, 16, 3, 2, 0, bias=False)
        self.bn1 = ComplexBatchNorm(16)
        self.relu = CReLU()

        self.conv2 = ComplexConv1d(16, 32, 3, bias=False)
        self.bn2 = ComplexBatchNorm(32)

        self.block1 = Block(32, 64, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(64, 128, 2, 2, start_with_relu=True, grow_first=True)

        self.block3 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block4 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)

        self.block10 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = CDSC1d(256, 512, 3, 1, 1)
        self.bn3 = ComplexBatchNorm(512)

        self.avgpool = ComplexAvgPool2d((1, 1))
        self.fc = ComplexLinear(512, self.num_classes)

        # ------- init weights --------
        for m in self.modules():
            # print(m, flush=True)
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.block10(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.view(x.size(0), x.size(1), x.size(2), 1)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

