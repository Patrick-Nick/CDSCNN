
"""
@FileName: train.py
@Author: Chenghong Xiao
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

def train_one_epoch(net, device, train_dataloader, optimizer, loss_func, LOSS, AC):
    """ training for one epoch
    Args:
        net: the classification model
        device: whether to use GPU
        train_dataloader: the training data loader
        optimizer: the optimization algorithm
        loss_func: the loss function
        LOSS: the list for recording the average loss for each epoch
        AC:  the list for recording the average accuracy for each epoch
    """
    net.train()
    Loss = 0
    ac = 0
    for i, (x, y) in enumerate(train_dataloader):
        x = x.float().to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = loss_func(out, y)
        Loss += loss
        loss.backward()
        optimizer.step()
        ac += accuracy_score(y.cpu().data.numpy(), torch.max(out, 1)[1].cpu().data.numpy())
    LOSS.append(Loss / (i + 1))
    AC.append(ac / (i + 1))

