import torch.nn as nn
import torch.nn.functional as F


class non_local(nn.Module):
    def __init__(self, dim, pool=True, active='num'):
        super(non_local, self).__init__()
        dim2 = 64
        self.dim2 = dim2
        self.q = nn.Conv2d(dim, dim2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.k = nn.Conv2d(dim, dim2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.v = nn.Conv2d(dim, dim2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.proj = nn.Conv2d(dim2, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        if pool:
            self.v = nn.Sequential(self.v, nn.MaxPool2d(kernel_size=(2, 2)))
            self.k = nn.Sequential(self.k, nn.AvgPool2d(kernel_size=(2, 2)))
        self.active = active

    def forward(self, x):
        b, c, h, w = x.shape
        v = self.v(x).reshape(b, self.dim2, -1).permute(0, 2, 1)  # b n2 c
        q = self.q(x).reshape(b, self.dim2, -1).permute(0, 2, 1)  # b n1 c
        k = self.k(x).reshape(b, self.dim2, -1)  # b c n2
        a = q @ k  # b n1 n2
        if self.active == 'softmax':
            a = F.softmax(a, dim=-1)
        elif self.active == 'num':
            a = a / a.size(-1)
        y = a @ v
        y = y.permute(0, 2, 1).contiguous().reshape(b, self.dim2, h, w)
        z = self.bn(self.proj(y)) + x
        return z





