import os
import glob
import re
from collections import defaultdict
import random


class msmt(object):
    def __init__(self):
        self.data_dir = '/home/lixulin/data/MSMT17_V1/'
        # 2815_c7_0004.jpg
        # pid cid relabel begin from 0, camid begin from 1
        dir_list1 = os.path.join(self.data_dir, 'list_train.txt')
        dir_list2 = os.path.join(self.data_dir, 'list_val.txt')
        dir_list3 = os.path.join(self.data_dir, 'list_query.txt')
        dir_list4 = os.path.join(self.data_dir, 'list_gallery.txt')
        with open(dir_list1, 'rt') as f:
            data_file_list = f.read().splitlines()
            self.img_paths1 = [os.path.join(self.data_dir, 'train', s.split(' ')[0]) for s in data_file_list]
        with open(dir_list2, 'rt') as f:
            data_file_list = f.read().splitlines()
            self.img_paths2 = [os.path.join(self.data_dir, 'train', s.split(' ')[0]) for s in data_file_list]
        with open(dir_list3, 'rt') as f:
            data_file_list = f.read().splitlines()
            self.img_paths3 = [os.path.join(self.data_dir, 'test', s.split(' ')[0]) for s in data_file_list]
        with open(dir_list4, 'rt') as f:
            data_file_list = f.read().splitlines()
            self.img_paths4 = [os.path.join(self.data_dir, 'test', s.split(' ')[0]) for s in data_file_list]
        print(f"=> loaded msmt17v1")
        print(f"  --------------------------------------------------------------------")
        print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
        print(f"  --------------------------------------------------------------------")
        self._info('train')
        self._info('val')
        self._info('query')
        self._info('gallery')
        self._info('all')
        print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c = self._process('train')
        self.query = self._process('query')
        self.gallery = self._process('gallery')

    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        if mode == 'train':
            img_paths = self.img_paths1
        if mode == 'val':
            img_paths = self.img_paths2
        if mode == 'query':
            img_paths = self.img_paths3
        if mode == 'gallery':
            img_paths = self.img_paths4
        if mode == 'all':
            img_paths = self.img_paths1 + self.img_paths2 + self.img_paths3 + self.img_paths4
        pattern = re.compile(r'(train|test)/\d+/(\d+)_\d+_(\d+)_\d+')
        img = 0
        for img_path in sorted(img_paths):
            imgtype = pattern.search(img_path).groups()[0]
            pid, camid = map(int, pattern.search(img_path).groups()[1:])
            if imgtype != 'train':#test id also begin from 0
                pid += 10000
            cid = 1
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
            img_paths = self.img_paths1 + self.img_paths2
        if mode == 'query':
            img_paths = self.img_paths3
        if mode == 'gallery':
            img_paths = self.img_paths4
        pattern = re.compile(r'\d+/(\d+)_\d+_(\d+)_\d+')
        if mode == 'train':
            pid_container = set()
            pid_cid_container = set()
            for img_path in sorted(img_paths):
                pid, _ = map(int, pattern.search(img_path).groups())
                cid = 1
                pid_container.add(pid)
                pid_cid_container.add((pid, cid))
            pid_container = sorted(pid_container)
            pid_cid_container = sorted(pid_cid_container)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            cid, mid = 1, 1
            if mode == 'train':
                cid = cid2label[pid, cid]
                pid = pid2label[pid]
            dataset.append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container)
        else:
            return dataset


if __name__ == '__main__':
    msmt()