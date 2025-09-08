import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class saidloss(nn.Module):
    def __init__(self, num_p, num_c, cid2pid， hdw=False):
        super(saidloss, self).__init__()
        self.num_p = num_p
        self.num_c = num_c
        self.cid2pid = cid2pid
        self.hdw = hdw

    def forward(self, y, pids, cids, mids, task):
        if task in ['vmsc', 'vmcc']:
            y = y[mids == 1]
            pids = pids[mids == 1]
            cids = cids[mids == 1]
        if task in ['imsc', 'imcc']:
            y = y[mids == 2]
            pids = pids[mids == 2]
            cids = cids[mids == 2]
        if task in ['vmcc', 'imcc', 'cmcc']:
            y_logsoftmax = F.log_softmax(y, dim=-1)
            pids_onehot = F.one_hot(pids, y.shape[1]).float()
            loss = -(pids_onehot * y_logsoftmax).sum(dim=-1)
        if task in ['vmsc', 'imsc', 'cmsc']:
            cids_cls = torch.tensor([range(self.num_c)]).type_as(cids)
            pids_cls = torch.tensor([self.cid2pid[i] for i in range(self.num_c)]).type_as(pids)
            n, m = y.shape
            mask1 = pids.expand(m, n).t().eq(pids_cls.expand(n, m))
            mask2 = cids.expand(m, n).t().ne(cids_cls.expand(n, m))
            mask = mask1 * mask2
            y = y * (mask == 0) - mask * 10
            y_logsoftmax = F.log_softmax(y, dim=-1)
            cids_onehot = F.one_hot(cids, y.shape[1]).float()
            loss = -(cids_onehot * y_logsoftmax).sum(dim=-1)
        return loss
