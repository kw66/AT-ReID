import os
import glob
import re
from collections import defaultdict
import random


class vc(object):
    def __init__(self):
        self.data_dir = '/home/lixulin/data/vc/'
        # pid-camid-cid-imgid
        # 0227-02-01-05.jpg
        # cam 2,3 same clothes 1, cam 1 clothes 1,2, cam 4 clothes 1,2,3
        # pid cid relabel begin from 0, camid begin from 1
        # test same(cam2,3)/cross(cam3,4)/cross2(cam all)
        print(f"=> loaded vc")
        print(f"  --------------------------------------------------------------------")
        print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
        print(f"  --------------------------------------------------------------------")
        self._info('train')
        self._info('query')
        self._info('gallery')
        self._info('all')
        print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c = self._process('train')
        self.query, self.query_cc, self.query_sc = self._process('query')
        self.gallery, self.gallery_cc, self.gallery_sc = self._process('gallery')

    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        img_paths1 = glob.glob(os.path.join(self.data_dir, 'train/*.[jp][pn]g'))
        img_paths2 = glob.glob(os.path.join(self.data_dir, 'query/*.[jp][pn]g'))
        img_paths3 = glob.glob(os.path.join(self.data_dir, 'gallery/*.[jp][pn]g'))
        if mode == 'train':
            img_paths = img_paths1
        if mode == 'query':
            img_paths = img_paths2
        if mode == 'gallery':
            img_paths = img_paths3
        if mode == 'all':
            img_paths = img_paths1 + img_paths2 + img_paths3
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-\d+')
        img = 0
        for img_path in sorted(img_paths):
            pid, camid, cid = map(int, pattern.search(img_path).groups())
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
            img_paths = glob.glob(os.path.join(self.data_dir, 'train/*.[jp][pn]g'))
        if mode == 'query':
            img_paths = glob.glob(os.path.join(self.data_dir, 'query/*.[jp][pn]g'))
        if mode == 'gallery':
            img_paths = glob.glob(os.path.join(self.data_dir, 'gallery/*.[jp][pn]g'))
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-\d+')
        if mode == 'train':
            pid_container = set()
            pid_cid_container = set()
            for img_path in sorted(img_paths):
                pid, _, cid = map(int, pattern.search(img_path).groups())
                #if camid in [2, 3]:
                #    continue
                pid_container.add(pid)
                pid_cid_container.add((pid, cid))
            pid_container = sorted(pid_container)
            pid_cid_container = sorted(pid_cid_container)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid, cid = map(int, pattern.search(img_path).groups())
            mid = 1
            if mode == 'train':
                #if camid in [2, 3]:
                #    continue
                cid = cid2label[pid, cid]
                pid = pid2label[pid]
            dataset.append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container)
        else:
            cross, same = [], []
            for index, (_, pid, cid, mid, camid) in enumerate(dataset):
                if camid in [3, 4]:
                    cross.append(dataset[index])
                if camid in [2, 3]:
                    same.append(dataset[index])
            return dataset, cross, same


if __name__ == '__main__':
    vc()