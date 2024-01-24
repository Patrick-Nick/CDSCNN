
"""
@FileName: validation.py
@Author: Chenghong Xiao
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


def val(net, device, val_dataloader, loss_func, val_LOSS, val_AC):
    """ validating for one epoch
        Args:
            net: the classification model
            device: whether to use GPU
            val_dataloader: the validation data loader
            loss_func: the loss function
            val_LOSS: the list for recording the average loss for each epoch
            val_AC: the list for recording the average accuracy for each epoch
    """
    net.eval()
    val_Loss = 0
    val_ac = 0
    with torch.no_grad():
        for i, (val_x, val_y) in enumerate(val_dataloader):
            val_x = val_x.float().to(device)
            val_y = val_y.long().to(device)
            val_out = net(val_x)
            val_loss = loss_func(val_out, val_y)
            val_Loss += val_loss
            val_ac += accuracy_score(val_y.cpu().data.numpy(), torch.max(val_out, 1)[1].cpu().data.numpy())

        val_LOSS.append(val_Loss / (i + 1))
        val_AC.append(val_ac / (i + 1))
