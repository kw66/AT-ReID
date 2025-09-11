import torch
from torch import nn


def compute_dist_eu(x1, x2):
    dist = torch.pow(x1, 2).sum(dim=-1, keepdim=True) + \
           torch.pow(x2, 2).sum(dim=-1, keepdim=True).t()
    dist.addmm_(x1, x2.t(), beta=1, alpha=-2)
    dist = dist.clamp(1e-12).sqrt()
    return dist


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