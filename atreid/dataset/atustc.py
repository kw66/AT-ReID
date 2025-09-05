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
            self._info('q_dtst')
            self._info('q_dtlt')
            self._info('q_ntst')
            self._info('q_ntlt')
            self._info('q_adst')
            self._info('q_adlt')
            self._info('g_dtst')
            self._info('g_dtlt')
            self._info('g_ntst')
            self._info('g_ntlt')
            self._info('g_adst')
            self._info('g_adlt')
            print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c, self.cid2pid = self._process('train')
        self.q_dtst = self._process('q_dtst')
        self.q_dtlt = self._process('q_dtlt')
        self.q_ntst = self._process('q_ntst')
        self.q_ntlt = self._process('q_ntlt')
        self.q_adst = self._process('q_adst')
        self.q_adlt = self._process('q_adlt')
        self.g_dtst = self._process('g_dtst')
        self.g_dtlt = self._process('g_dtlt')
        self.g_ntst = self._process('g_ntst')
        self.g_ntlt = self._process('g_ntlt')
        self.g_adst = self._process('g_adst')
        self.g_adlt = self._process('g_adlt')

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
            if mode == 'q_dtst' and imtype != 2:
                continue
            if mode == 'q_dtlt' and imtype != 3:
                continue
            if mode == 'q_ntst' and imtype != 4:
                continue
            if mode == 'q_ntlt' and imtype != 5:
                continue
            if mode == 'q_adst' and imtype not in [2, 4]:
                continue
            if mode == 'q_adlt' and imtype not in [3, 5]:
                continue
            if mode == 'g_dtst' and imtype != 6:
                continue
            if mode == 'g_dtlt' and imtype != 7:
                continue
            if mode == 'g_ntst' and imtype != 8:
                continue
            if mode == 'g_ntlt' and imtype != 9:
                continue
            if mode == 'g_adst' and imtype not in [6, 8]:
                continue
            if mode == 'g_adlt' and imtype not in [7, 9]:
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
            if mode == 'q_dtst' and imtype != 2:
                continue
            if mode == 'q_dtlt' and imtype != 3:
                continue
            if mode == 'q_ntst' and imtype != 4:
                continue
            if mode == 'q_ntlt' and imtype != 5:
                continue
            if mode == 'q_adst' and imtype not in [2, 4]:
                continue
            if mode == 'q_adlt' and imtype not in [3, 5]:
                continue
            if mode == 'g_dtst' and imtype != 6:
                continue
            if mode == 'g_dtlt' and imtype != 7:
                continue
            if mode == 'g_ntst' and imtype != 8:
                continue
            if mode == 'g_ntlt' and imtype != 9:
                continue
            if mode == 'g_adst' and imtype not in [6, 8]:
                continue
            if mode == 'g_adlt' and imtype not in [7, 9]:
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
