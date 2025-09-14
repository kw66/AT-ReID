import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class PSKSampler(Sampler):
    def __init__(self, data_source, num_p=8, num_k=8, nn=100, sample_mode='p'):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_p = num_p
        self.num_k = num_k
        self.sample_mode = sample_mode
        self.p = defaultdict(set)
        if self.sample_mode == 'p':
            self.index_dic = defaultdict(list)
        if self.sample_mode in ['c', 'm', 'cam']:
            self.index_dic = defaultdict(lambda: defaultdict(list))
        if self.sample_mode in ['mcam', 'ccam', 'mc', 'camm', 'camc']:
            self.index_dic = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, (_, pid, cid, mid, camid) in enumerate(self.data_source):
            if self.sample_mode == 'p':
                self.index_dic[pid].append(index)
            if self.sample_mode == 'c':
                self.index_dic[pid][cid].append(index)
            if self.sample_mode == 'm':
                self.index_dic[pid][mid].append(index)
            if self.sample_mode == 'cam':
                self.index_dic[pid][camid].append(index)
            if self.sample_mode == 'mcam':
                self.index_dic[pid][mid][camid].append(index)
            if self.sample_mode == 'ccam':
                self.index_dic[pid][cid][camid].append(index)
            if self.sample_mode == 'mc':
                self.index_dic[pid][mid][cid].append(index)
            if self.sample_mode == 'camm':
                self.index_dic[pid][camid][mid].append(index)
            if self.sample_mode == 'camc':
                self.index_dic[pid][camid][cid].append(index)
            self.p[pid].add((cid, mid, camid))
        self.pids = sorted(self.index_dic.keys())
        self.p = np.array([float(len(self.p[pid])) for pid in self.pids])
        self.p /= sum(self.p)
        self.length = self.num_p * self.num_k * nn

    def __iter__(self):
        ret = []
        while len(ret) < self.length:
            selected_pids = np.random.choice(self.pids, self.num_p, replace=False,)
            for pid in sorted(selected_pids):
                if self.sample_mode == 'p':
                    idxs = self.index_dic[pid]
                    num_k = int(self.num_k)
                    ret.extend(np.random.choice(idxs, num_k, replace=len(idxs) < num_k))
                if self.sample_mode in ['c', 'm', 'cam']:
                    sids = list(self.index_dic[pid].keys())
                    num_s = min(len(sids), 2)
                    selected_sids = np.random.choice(sids, num_s, replace=False)
                    for sid in sorted(selected_sids):
                        idxs = self.index_dic[pid][sid]
                        num_k = int(self.num_k / num_s)
                        ret.extend(np.random.choice(idxs, num_k, replace=len(idxs) < num_k))
                if self.sample_mode in ['mcam', 'ccam', 'mc', 'camm', 'camc']:
                    s1ids = list(self.index_dic[pid].keys())
                    num_s1 = min(len(s1ids), 2)
                    selected_s1ids = np.random.choice(s1ids, num_s1, replace=False)
                    for s1id in sorted(selected_s1ids):
                        s2ids = list(self.index_dic[pid][s1id].keys())
                        num_s2 = min(len(s2ids), 2)
                        selected_s2ids = np.random.choice(s2ids, num_s2, replace=False)
                        for s2id in sorted(selected_s2ids):
                            idxs = self.index_dic[pid][s1id][s2id]
                            num_k = int(self.num_k / num_s1 / num_s2)
                            ret.extend(np.random.choice(idxs, num_k, replace=len(idxs) < num_k))
        return iter(ret)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass
