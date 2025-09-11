import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from loss.said import saidloss


class saidloss_hdw(nn.Module):
    def __init__(self, cid2pid, said=False, hdw=False):
        super(saidloss_hdw, self).__init__()
        self.cid2pid = cid2pid
        self.said = said
        self.hdw = hdw

    def forward(self, y, pids, cids, mids):
        l_dt_st = saidloss(y[0], pids, cids, mids, self.cid2pid, 'dt-st', self.said, self.hdw)
        l_dt_lt = saidloss(y[1], pids, cids, mids, self.cid2pid, 'dt-lt', self.said, self.hdw)
        l_nt_st = saidloss(y[2], pids, cids, mids, self.cid2pid, 'nt-st', self.said, self.hdw)
        l_nt_lt = saidloss(y[3], pids, cids, mids, self.cid2pid, 'nt-lt', self.said, self.hdw)
        l_ad_st = saidloss(y[4], pids, cids, mids, self.cid2pid, 'ad-st', self.said, self.hdw)
        l_ad_lt = saidloss(y[5], pids, cids, mids, self.cid2pid, 'ad-lt', self.said, self.hdw)
        if self.hdw:
            b = pids.shape[0] * 1.0
            p_dt_st = (-l_dt_st).exp()
            p_dt_lt = (-l_dt_lt).exp()
            p_nt_st = (-l_nt_st).exp()
            p_nt_lt = (-l_nt_lt).exp()
            p_ad_st = (-l_ad_st).exp()
            p_ad_lt = (-l_ad_lt).exp()
            w_dt = (1 - torch.cat((p_dt_st, p_dt_lt), 0).mean()) ** 0.5
            w_nt = (1 - torch.cat((p_nt_st, p_nt_lt), 0).mean()) **0.5
            w_ad = (1 - torch.cat((p_ad_st, p_ad_lt), 0).mean()) **0.5
            w_st = (1 - torch.cat((p_dt_st, p_nt_st, p_ad_st), 0).mean()) ** 0.5
            w_lt = (1 - torch.cat((p_dt_lt, p_nt_lt, p_ad_lt), 0).mean()) ** 0.5
            w_tm = max(w_dt, w_nt, w_ad).clamp(1e-6)
            w_ti = max(w_st, w_lt).clamp(1e-6)
            l_dt_st = (w_dt / w_tm) * (w_st / w_ti) * l_dt_st
            l_dt_lt = (w_dt / w_tm) * (w_lt / w_ti) * l_dt_lt
            l_nt_st = (w_nt / w_tm) * (w_st / w_ti) * l_nt_st
            l_nt_lt = (w_nt / w_tm) * (w_lt / w_ti) * l_nt_lt
            l_ad_st = (w_ad / w_tm) * (w_st / w_ti) * l_ad_st
            l_ad_lt = (w_ad / w_tm) * (w_lt / w_ti) * l_ad_lt
            loss = l_dt_st.sum() + l_dt_lt.sum() + l_nt_st.sum() + l_nt_lt.sum() + l_ad_st.sum() + l_ad_lt.sum()
            loss = loss / b
        else:
            loss = l_dt_st + l_dt_lt + l_nt_st + l_nt_lt + l_ad_st + l_ad_lt
        return loss
