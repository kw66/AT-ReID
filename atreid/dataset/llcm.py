import os
import glob
import re
from collections import defaultdict
import random
import numpy as np


class llcm(object):
    def __init__(self):
        self.data_dir = '/home/lixulin/data/llcm/'
        # 0006_c3_0001.jpg
        # pid cid relabel begin from 0, camid begin from 1
        print(f"=> loaded llcm")
        print(f"  --------------------------------------------------------------------")
        print(f"   subset   |  ids | clothes | rgb_cam | ir_cam | clothes_cam | images ")
        print(f"  --------------------------------------------------------------------")
        self._info('train_rgb')
        self._info('train_ir')
        self._info('test_rgb')
        self._info('test_ir')
        self._info('all')
        print(f"  --------------------------------------------------------------------")
        self.train, self.num_p, self.num_c = self._process('train')
        self.q_rgb, self.g_rgb = self._process('test_rgb')
        self.q_ir, self.g_ir = self._process('test_ir')

    def _info(self, mode):
        pids = set()
        p_cids = set()
        p_rgb_camids = set()
        p_ir_camids = set()
        p_c_camids = set()
        img_paths1 = glob.glob(os.path.join(self.data_dir, 'train_rgb/*.[jp][pn]g'))
        img_paths2 = glob.glob(os.path.join(self.data_dir, 'train_ir/*.[jp][pn]g'))
        img_paths3 = glob.glob(os.path.join(self.data_dir, 'test_rgb/*.[jp][pn]g'))
        img_paths4 = glob.glob(os.path.join(self.data_dir, 'test_ir/*.[jp][pn]g'))
        if mode == 'train_rgb':
            img_paths = img_paths1
        if mode == 'train_ir':
            img_paths = img_paths2
        if mode == 'test_rgb':
            img_paths = img_paths3
        if mode == 'test_ir':
            img_paths = img_paths4
        if mode == 'all':
            img_paths = img_paths1 + img_paths2 + img_paths3 + img_paths4
        pattern = re.compile(r'(\d+)_c(\d+)')
        img = 0
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            cid = 1
            mid = 1 if camid <= 9 else 2
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
        #print(len(p_rgb_camids))
        rgb = len(p_rgb_camids) / len(pids)
        ir = len(p_ir_camids) / len(pids)
        c_cam = len(p_c_camids) / len(pids)
        print(f"   {mode:9s}| {ids:4d} | {c:7.1f} | {rgb:7.1f} | {ir:6.1f} | {c_cam:11.1f} | {img:6d} ")

    def _process(self, mode):
        if mode == 'train':
            img_paths = glob.glob(os.path.join(self.data_dir, 'train/*.[jp][pn]g'))
        if mode == 'test_rgb':
            img_paths = glob.glob(os.path.join(self.data_dir, 'test_rgb/*.[jp][pn]g'))
        if mode == 'test_ir':
            img_paths = glob.glob(os.path.join(self.data_dir, 'test_ir/*.[jp][pn]g'))
        pattern = re.compile(r'(\d+)_c(\d+)')
        if mode == 'train':
            pid_container = set()
            pid_cid_container = set()
            for img_path in sorted(img_paths):
                pid, camid = map(int, pattern.search(img_path).groups())
                cid = 1
                #mid = 1 if camid <= 9 else 2
                #if mid == 2:
                #    continue
                pid_container.add(pid)
                pid_cid_container.add((pid, cid))
            pid_container = sorted(pid_container)
            pid_cid_container = sorted(pid_cid_container)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            cid2label = {(pid, cid): label for label, (pid, cid) in enumerate(pid_cid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            cid = 1
            mid = 1 if camid <= 9 else 2
            #if mid == 2:
            #    continue
            if mode == 'train':
                cid = cid2label[pid, cid]
                pid = pid2label[pid]
            dataset.append((img_path, pid, cid, mid, camid))
        if mode == 'train':
            return dataset, len(pid_container), len(pid_cid_container)
        else:
            index_dic_all = defaultdict(list)
            all_single = defaultdict(list)
            for index, (_, pid, cid, mid, camid) in enumerate(dataset):
                index_dic_all[pid, camid].append(index)
            x = 5
            if x == 0:
                for trial in range(10):
                    random.seed(trial)
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].append(dataset[random.choice(index_dic_all[pid_camid])])
            if x == 1:
                random.seed(0)
                for pid_camid in list(index_dic_all.keys()):
                    for trial in range(10):
                        all_single[trial].append(dataset[random.choice(index_dic_all[pid_camid])])
            if x == 2:
                random.seed(0)
                for pid_camid in list(index_dic_all.keys()):
                    index = random.sample(index_dic_all[pid_camid], k=10)  # sample 是相当于不放回抽样
                    for trial in range(10):
                        all_single[trial].append(dataset[index[trial]])
            if x == 3:
                random.seed(0)
                for pid_camid in list(index_dic_all.keys()):
                    index = random.choices(index_dic_all[pid_camid], k=10)
                    for trial in range(10):
                        all_single[trial].append(dataset[index[trial]])
            if x == 4:
                for trial in range(10):
                    random.seed(trial + 1)
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].append(dataset[random.choice(index_dic_all[pid_camid])])
            if x == 5:
                for trial in range(10):
                    np.random.seed(trial)
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].append(dataset[np.random.choice(index_dic_all[pid_camid])])
            if x == 6:
                for trial in range(10):
                    random.seed(trial)
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].extend(np.array(dataset)[random.sample(index_dic_all[pid_camid], k=1)])
            if x == 7:
                np.random.seed(0)
                for trial in range(10):
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].append(dataset[np.random.choice(index_dic_all[pid_camid])])
            if x == 8:
                for trial in range(10):
                    np.random.seed(trial + 1)
                    for pid_camid in list(index_dic_all.keys()):
                        all_single[trial].append(dataset[np.random.choice(index_dic_all[pid_camid])])
            return dataset, all_single


if __name__ == '__main__':
    llcm()