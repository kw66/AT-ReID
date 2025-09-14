import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def saidloss(y, pids, cids, mids, cid2pid, scenario, said=False, hdw=False):
    if not said:
        scenario = 'ad-lt'
    if scenario in ['dt-st', 'dt-lt']:
        y = y[mids == 1]
        pids = pids[mids == 1]
        cids = cids[mids == 1]
    if scenario in ['nt-st', 'nt-lt']:
        y = y[mids == 2]
        pids = pids[mids == 2]
        cids = cids[mids == 2]
    if scenario in ['dt-lt', 'nt-lt', 'ad-lt']:
        y_logsoftmax = F.log_softmax(y, dim=-1)
        if not hdw:
            loss = F.nll_loss(y_logsoftmax, pids)
        else:
            pids_onehot = F.one_hot(pids, y.shape[1]).float()
            loss = -(pids_onehot * y_logsoftmax).sum(dim=-1)
    if scenario in ['dt-st', 'nt-st', 'ad-st']:
        num_c = len(cid2pid)
        cids_cls = torch.tensor([range(num_c)]).type_as(cids)
        pids_cls = torch.tensor([cid2pid[i] for i in range(num_c)]).type_as(pids)
        n, m = y.shape
        mask1 = pids.expand(m, n).t().eq(pids_cls.expand(n, m))
        mask2 = cids.expand(m, n).t().ne(cids_cls.expand(n, m))
        mask = mask1 * mask2
        y = y * (mask == 0) - mask * 10
        y_logsoftmax = F.log_softmax(y, dim=-1)
        if not hdw:
            loss = F.nll_loss(y_logsoftmax, cids)
        else:
            cids_onehot = F.one_hot(cids, y.shape[1]).float()
            loss = -(cids_onehot * y_logsoftmax).sum(dim=-1)
    return loss