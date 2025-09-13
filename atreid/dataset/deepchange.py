import os
import glob
import re
from collections import defaultdict
import random


class deepchange(object):
    def __init__(self):
        self.data_dir = '/home/lixulin/data/deepchange/'
        # /train-set/p0702_c13_20200813_154757_000038_bbox.jpg
        # pid_camid_time_trajectory time as clothes
        print(f"=> loaded deepchange")
        print(f"  --------------------------------------------------------------------")
        print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
        print(f"  --------------------------------------------------------------------")
        self._info('train')
        self._info('val')
        self._info('query')
        self._info('gallery')
        self._info('all')
        print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c, self.cid2pid = self._process('train')
        self.query = self._process('query')
        self.gallery = self._process('gallery')

    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        img_paths1 = glob.glob(os.path.join(self.data_dir, 'train-set/*.[jp][pn]g'))
        img_paths2 = glob.glob(os.path.join(self.data_dir, 'val-set-query/*.[jp][pn]g'))
        img_paths3 = glob.glob(os.path.join(self.data_dir, 'val-set-gallery/*.[jp][pn]g'))
        img_paths4 = glob.glob(os.path.join(self.data_dir, 'test-set-query/*.[jp][pn]g'))
        img_paths5 = glob.glob(os.path.join(self.data_dir, 'test-set-gallery/*.[jp][pn]g'))
        if mode == 'train':
            img_paths = img_paths1
        if mode == 'val':
            img_paths = img_paths2 + img_paths3
        if mode == 'query':
            img_paths = img_paths4
        if mode == 'gallery':
            img_paths = img_paths5
        if mode == 'all':
            img_paths = img_paths1 + img_paths2 + img_paths3 + img_paths4 + img_paths5
        pattern = re.compile(r'(\d+)_c(\d+)_(\d+)_(\d+)')
        img = 0
        for img_path in sorted(img_paths):
            pid, camid, cid, _ = map(int, pattern.search(img_path).groups())
            mid = 1
            pids.add(pid)
            p_cids.add((pid, cid))
            if mid == 1:
                p_rgb_camids.add((pid, camid))
            if mid == 2:
                p_ir_camids.add((pid, camid))
            p_c_camids.add((pid, cid, camid))
            img += 1
        ids = len(pids)
        c = len(p_cids) / len(pids)
        rgb = len(p_rgb_camids) / len(pids)
        ir = len(p_ir_camids) / len(pids)
        c_cam = len(p_c_camids) / len(pids)
        print(f"   {mode:9s}| {ids:4d} | {c:7.1f} | {rgb:7.1f} | {ir:6.1f} | {c_cam:11.1f} | {img:6d} ")

    def _process(self, mode):
        if mode == 'train':
            img_paths = glob.glob(os.path.join(self.data_dir, 'train-set/*.[jp][pn]g'))# + \
                        #glob.glob(os.path.join(self.data_dir, 'val-set-query/*.[jp][pn]g')) + \
                        #glob.glob(os.path.join(self.data_dir, 'val-set-gallery/*.[jp][pn]g'))
        if mode == 'query':
            img_paths = glob.glob(os.path.join(self.data_dir, 'test-set-query/*.[jp][pn]g'))
        if mode == 'gallery':
            img_paths = glob.glob(os.path.join(self.data_dir, 'test-set-gallery/*.[jp][pn]g'))
        pattern = re.compile(r'(\d+)_c(\d+)_(\d+)_(\d+)')
        if mode == 'train':
            pid_container = set()
            pid_cid_container = set()
            cid2pid = {}
            for img_path in sorted(img_paths):
                pid, _, cid, _ = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
                pid_cid_container.add((pid, cid))
            pid_container = sorted(pid_container)
            pid_cid_container = sorted(pid_cid_container)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid, cid, cid2 = map(int, pattern.search(img_path).groups())
            mid = 1
            if mode == 'train':
                cid = cid2label[pid, cid]
                pid = pid2label[pid]
                cid2pid[cid] = pid
            if mode == 'query' or mode == 'gallery':
                cid = cid2 + (cid % 10000) * 1000000
            dataset.append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container), cid2pid
        else:
            return dataset


if __name__ == '__main__':
    deepchange()