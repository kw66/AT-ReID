import os
import glob
import re


class atustc(object):
    def __init__(self, info=True):
        self.data_dir = '/home/lixulin/data/atustc/'
        # p001-d01-c01/cam01-f0-1060.jpg
        # pid cid relabel begin from 0, camid begin from 1
        if info:
            print(f"=> loaded atustc")
            print(f"  --------------------------------------------------------------------")
            print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
            print(f"  --------------------------------------------------------------------")
            self._info('train')
            self._info('train_rgb')
            self._info('train_ir')
            self._info('val')
            self._info('test')
            self._info('all')
            print(f"  --------------------------------------------------------------------")
            self._info('q_dt_st')
            self._info('q_dt_lt')
            self._info('q_nt_st')
            self._info('q_nt_lt')
            self._info('q_ad_st')
            self._info('q_ad_lt')
            self._info('g_dt_st')
            self._info('g_dt_lt')
            self._info('g_nt_st')
            self._info('g_nt_lt')
            self._info('g_ad_st')
            self._info('g_ad_lt')
            print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c, self.cid2pid = self._process('train')
        self.q_dt_st = self._process('q_dt_st')
        self.q_dt_lt = self._process('q_dt_lt')
        self.q_nt_st = self._process('q_nt_st')
        self.q_nt_lt = self._process('q_nt_lt')
        self.q_ad_st = self._process('q_ad_st')
        self.q_ad_lt = self._process('q_ad_lt')
        self.g_dt_st = self._process('g_dt_st')
        self.g_dt_lt = self._process('g_dt_lt')
        self.g_nt_st = self._process('g_nt_st')
        self.g_nt_lt = self._process('g_nt_lt')
        self.g_ad_st = self._process('g_ad_st')
        self.g_ad_lt = self._process('g_ad_lt')

    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        img_paths = glob.glob(os.path.join(self.data_dir, '*/*.[jp][pn]g'))
        pattern = re.compile(r'p(\d+)-d(\d+)-c(\d+)/cam(\d+)-f(\d+)-\d+')
        img = 0
        for img_path in sorted(img_paths):
            pid, did, cid, camid, imtype = map(int, pattern.search(img_path).groups())
            mid = 1 if camid <= 8 else 2
            if mode == 'train' and imtype != 0:
                continue
            if mode == 'train_rgb' and (imtype != 0 or mid != 1):
                continue
            if mode == 'train_ir' and (imtype != 0 or mid != 2):
                continue
            if mode == 'val' and imtype != 1:
                continue
            if mode == 'test' and imtype in [0, 1]:
                continue
            if mode == 'q_dt_st' and imtype != 2:
                continue
            if mode == 'q_dt_lt' and imtype != 3:
                continue
            if mode == 'q_nt_st' and imtype != 4:
                continue
            if mode == 'q_nt_lt' and imtype != 5:
                continue
            if mode == 'q_ad_st' and imtype not in [2, 4]:
                continue
            if mode == 'q_ad_lt' and imtype not in [3, 5]:
                continue
            if mode == 'g_dt_st' and imtype != 6:
                continue
            if mode == 'g_dt_lt' and imtype != 7:
                continue
            if mode == 'g_nt_st' and imtype != 8:
                continue
            if mode == 'g_nt_lt' and imtype != 9:
                continue
            if mode == 'g_ad_st' and imtype not in [6, 8]:
                continue
            if mode == 'g_ad_lt' and imtype not in [7, 9]:
                continue
            pids.add(pid)
            p_cids.add((pid, cid))
            if mid == 1:
                p_rgb_camids.add((pid, camid))
            if mid == 2:
                p_ir_camids.add((pid, camid))
            p_c_camids.add((pid, cid, camid))
            img += 1
        ids = len(pids)
        c = len(p_cids) / max(1, len(pids))
        rgb = len(p_rgb_camids) / max(1, len(pids))
        ir = len(p_ir_camids) / max(1, len(pids))
        c_cam = len(p_c_camids) / max(1, len(pids))
        print(f"   {mode:9s}| {ids:4d} | {c:7.1f} | {rgb:7.1f} | {ir:6.1f} | {c_cam:11.1f} | {img:6d} ")

    def _process(self, mode):
        img_paths = glob.glob(os.path.join(self.data_dir, '*/*.[jp][pn]g'))
        pattern = re.compile(r'p(\d+)-d(\d+)-c(\d+)/cam(\d+)-f(\d+)-\d+')
        if mode == 'train':
            pid_container = set()
            pid_cid_container = set()
            cid2pid = {}
            for img_path in sorted(img_paths):
                pid, did, cid, camid, imtype = map(int, pattern.search(img_path).groups())
                if imtype != 0:
                    continue
                pid_container.add(pid)
                pid_cid_container.add((pid, cid))
            pid_container = sorted(pid_container)
            pid_cid_container = sorted(pid_cid_container)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
        dataset = []
        for img_path in img_paths:
            pid, did, cid, camid, imtype = map(int, pattern.search(img_path).groups())
            mid = 1 if camid <= 8 else 2
            if mode == 'train' and imtype != 0:
                continue
            if mode == 'q_dt_st' and imtype != 2:
                continue
            if mode == 'q_dt_lt' and imtype != 3:
                continue
            if mode == 'q_nt_st' and imtype != 4:
                continue
            if mode == 'q_nt_lt' and imtype != 5:
                continue
            if mode == 'q_ad_st' and imtype not in [2, 4]:
                continue
            if mode == 'q_ad_lt' and imtype not in [3, 5]:
                continue
            if mode == 'g_dt_st' and imtype != 6:
                continue
            if mode == 'g_dt_lt' and imtype != 7:
                continue
            if mode == 'g_nt_st' and imtype != 8:
                continue
            if mode == 'g_nt_lt' and imtype != 9:
                continue
            if mode == 'g_ad_st' and imtype not in [6, 8]:
                continue
            if mode == 'g_ad_lt' and imtype not in [7, 9]:
                continue
            if mode == 'train':
                cid = cid2label[pid, cid]
                pid = pid2label[pid]
                cid2pid[cid] = pid
            dataset.append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container), cid2pid
        else:
            return dataset


if __name__ == '__main__':
    atustc()
