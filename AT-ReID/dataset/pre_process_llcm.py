import os
import glob
import re
import shutil
from tqdm import tqdm

dir = "/home/lixulin/data/llcm/"

if not os.path.isdir(dir+'train/'):
    os.makedirs(dir+'train/')
if not os.path.isdir(dir+'train_rgb/'):
    os.makedirs(dir+'train_rgb/')
if not os.path.isdir(dir+'train_ir/'):
    os.makedirs(dir+'train_ir/')
if not os.path.isdir(dir+'test_rgb/'):
    os.makedirs(dir+'test_rgb/')
if not os.path.isdir(dir+'test_ir/'):
    os.makedirs(dir+'test_ir/')

file_path_test = os.path.join(dir, 'idx/test_id.txt')
with open(file_path_test, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_test = ["%04d" % x for x in ids]
train_nir = os.path.join(dir, 'idx/train_nir.txt')
train_vis = os.path.join(dir, 'idx/train_vis.txt')
test_nir = os.path.join(dir, 'idx/test_nir.txt')
test_vis = os.path.join(dir, 'idx/test_vis.txt')
#"/home/lixulin/data/llcm/nir/0001/0001_c01_s185535_f7010_nir.jpg"
with open(train_vis, 'r') as f:
    pattern = re.compile(r'(\d+)_c(\d+)(_s\d+_f\d+_vis)')
    data_file_list = f.read().splitlines()
    for data_file in tqdm(data_file_list):
        img_path = os.path.join(dir, data_file.split(' ')[0])
        #pid = data_file.split(' ')[1]
        pid, camid, imgid = pattern.search(img_path).groups()
        if pid not in id_test:
            camid = '_c' + camid
            shutil.copyfile(img_path, dir + 'train_rgb/' + pid + camid + imgid + '.jpg')
            shutil.copyfile(img_path, dir + 'train/' + pid + camid + imgid + '.jpg')
with open(train_nir, 'r') as f:
    pattern = re.compile(r'(\d+)_c(\d+)(_s\d+_f\d+_nir)')
    data_file_list = f.read().splitlines()
    for data_file in tqdm(data_file_list):
        img_path = os.path.join(dir, data_file.split(' ')[0])
        #pid = data_file.split(' ')[1]
        pid, camid, imgid = pattern.search(img_path).groups()
        if pid not in id_test:
            camid = '_c' + str(int(camid) + 9)
            shutil.copyfile(img_path, dir + 'train_ir/' + pid + camid + imgid + '.jpg')
            shutil.copyfile(img_path, dir + 'train/' + pid + camid + imgid + '.jpg')

#"/home/lixulin/data/llcm/test_nir/cam1/0003/0003_c01_s185734_f3195_nir.jpg"
with open(test_vis, 'r') as f:
    pattern = re.compile(r'(\d+)_c(\d+)(_s\d+_f\d+_vis)')
    data_file_list = f.read().splitlines()
    for data_file in tqdm(data_file_list):
        img_path = os.path.join(dir, data_file.split(' ')[0])
        #pid = data_file.split(' ')[1]
        pid, camid, imgid = pattern.search(img_path).groups()
        if pid in id_test:
            camid = '_c' + camid
            shutil.copyfile(img_path, dir + 'test_rgb/' + pid + camid + imgid + '.jpg')
with open(test_nir, 'r') as f:
    pattern = re.compile(r'(\d+)_c(\d+)(_s\d+_f\d+_nir)')
    data_file_list = f.read().splitlines()
    for data_file in tqdm(data_file_list):
        img_path = os.path.join(dir, data_file.split(' ')[0])
        #pid = data_file.split(' ')[1]
        pid, camid, imgid = pattern.search(img_path).groups()
        if pid in id_test:
            camid = '_c' + str(int(camid) + 9)
            shutil.copyfile(img_path, dir + 'test_ir/' + pid + camid + imgid + '.jpg')

if __name__ == '__main__':
    pass
