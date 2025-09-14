import os
import glob
import re
from collections import defaultdict
import random


class regdb(object):
    def __init__(self, trial=1):
        self.data_dir = '/home/lixulin/data/regdb/'
        #male_front_v_00007_1.bmp
        #female_back_t_06300_2.bmp
        self.img_paths1 = defaultdict(list)
        self.img_paths2 = defaultdict(list)
        self.img_paths3 = defaultdict(list)
        self.img_paths4 = defaultdict(list)
        self.trial = trial
        for trial in range(1, 11):
            dir_list1 = os.path.join(self.data_dir, f'idx/train_visible_{trial}.txt')
            dir_list2 = os.path.join(self.data_dir, f'idx/train_thermal_{trial}.txt')
            dir_list3 = os.path.join(self.data_dir, f'idx/test_visible_{trial}.txt')
            dir_list4 = os.path.join(self.data_dir, f'idx/test_thermal_{trial}.txt')
            with open(dir_list1, 'rt') as f:
                data_file_list = f.read().splitlines()
                self.img_paths1[trial] = [os.path.join(self.data_dir, s.split(' ')[0]) for s in data_file_list]
            with open(dir_list2, 'rt') as f:
                data_file_list = f.read().splitlines()
                self.img_paths2[trial] = [os.path.join(self.data_dir, s.split(' ')[0]) for s in data_file_list]
            with open(dir_list3, 'rt') as f:
                data_file_list = f.read().splitlines()
                self.img_paths3[trial] = [os.path.join(self.data_dir, s.split(' ')[0]) for s in data_file_list]
            with open(dir_list4, 'rt') as f:
                data_file_list = f.read().splitlines()
                self.img_paths4[trial] = [os.path.join(self.data_dir, s.split(' ')[0]) for s in data_file_list]
        print(f"=> loaded regdb")
        print(f"  --------------------------------------------------------------------")
        print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
        print(f"  --------------------------------------------------------------------")
        self._info('train_rgb')
        self._info('train_ir')
        self._info('test_rgb')
        self._info('test_ir')
        self._info('all')
        print(f"  --------------------------------------------------------------------")
        self.train10, self.num_p, self.num_c = self._process('train')
        self.test_rgb = self._process('test_rgb')
        self.test_ir = self._process('test_ir')
        self.train = self.train10[self.trial]


    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        if mode == 'train_rgb':
            img_paths = self.img_paths1[1]
        if mode == 'train_ir':
            img_paths = self.img_paths2[1]
        if mode == 'test_rgb':
            img_paths = self.img_paths3[1]
        if mode == 'test_ir':
            img_paths = self.img_paths4[1]
        if mode == 'all':
            img_paths = self.img_paths1[1] + self.img_paths2[1] + \
                        self.img_paths3[1] + self.img_paths4[1]
        pattern = re.compile(r'([vt])_\d+_(\d+)')
        img = 0
        for img_path in sorted(img_paths):
            m, pid = pattern.search(img_path).groups()
            pid = int(pid)
            cid = 1
            mid = 1 if m == 'v' else 2
            camid = mid
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
        dataset = defaultdict(list)
        for trial in range(1, 11):
            if mode == 'train':
                img_paths = self.img_paths1[trial] + self.img_paths2[trial]
            if mode == 'test_rgb':
                img_paths = self.img_paths3[trial]
            if mode == 'test_ir':
                img_paths = self.img_paths4[trial]
            pattern = re.compile(r'([vt])_\d+_(\d+)')
            if mode == 'train':
                pid_container = set()
                pid_cid_container = set()
                for img_path in sorted(img_paths):
                    m, pid = pattern.search(img_path).groups()
                    pid = int(pid)
                    cid = 1
                    pid_container.add(pid)
                    pid_cid_container.add((pid, cid))
                pid_container = sorted(pid_container)
                pid_cid_container = sorted(pid_cid_container)
                pid2label = {pid: label for label, pid in enumerate(pid_container)}
                cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
            for img_path in sorted(img_paths):
                m, pid = pattern.search(img_path).groups()
                pid = int(pid)
                cid = 1
                mid = 1 if m == 'v' else 2
                camid = mid
                if mode == 'train':
                    cid = cid2label[pid, cid]
                    pid = pid2label[pid]
                dataset[trial].append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container)
        else:
            return dataset


if __name__ == '__main__':

    regdb()
