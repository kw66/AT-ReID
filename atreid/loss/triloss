import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, x, pids):
        dist = compute_dist_eu(x, x)
        n, m = dist.shape
        mask_p = pids.expand(n, m).eq(pids.expand(m, n).t())
        mask_n = pids.expand(n, m).ne(pids.expand(m, n).t())
        loss = ((((dist * mask_p).max(1)[0].unsqueeze(1) + self.margin - dist).clamp(0) * mask_n).max(1)[0]).mean()
        return loss
