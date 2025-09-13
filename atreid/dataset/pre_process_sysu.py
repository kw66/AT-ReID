import os
import glob
import re
import shutil
from tqdm import tqdm

dir = "/home/lixulin/data/sysu/"
if not os.path.isdir(dir+'train/'):
    os.makedirs(dir+'train/')
if not os.path.isdir(dir+'train_rgb/'):
    os.makedirs(dir+'train_rgb/')
if not os.path.isdir(dir+'train_ir/'):
    os.makedirs(dir+'train_ir/')
if not os.path.isdir(dir+'query/'):
    os.makedirs(dir+'query/')
if not os.path.isdir(dir+'gallery/'):
    os.makedirs(dir+'gallery/')
file_path_test = os.path.join(dir, 'exp/test_id.txt')
with open(file_path_test, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_test = ["%04d" % x for x in ids]
file_path_train = os.path.join(dir, 'exp/train_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
file_path_val = os.path.join(dir, 'exp/val_id.txt')
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
id_train += id_val
img_paths = glob.glob(dir + '*/*/*.[jp][pn]g')
pattern = re.compile(r'cam(\d+)/(\d+)/(\d+)')
for img_path in tqdm(img_paths):
    camid, pid, imgid = pattern.search(img_path).groups()
    if pid in id_train:
        if int(camid) in [1, 2, 4, 5]:
            camid = '_c' + camid + '_'
            shutil.copyfile(img_path, dir + 'train_rgb/' + pid + camid + imgid + '.jpg')
        elif int(camid) in [3,6]:
            camid = '_c' + camid + '_'
            shutil.copyfile(img_path, dir + 'train_ir/' + pid + camid + imgid + '.jpg')
        shutil.copyfile(img_path, dir + 'train/' + pid + camid + imgid + '.jpg')
    if pid in id_test:
        if camid in ['3', '6']:
            camid = '_c' + camid + '_'
            shutil.copyfile(img_path, dir + 'query/' + pid + camid + imgid + '.jpg')
        elif camid in ['1', '2', '4', '5']:
            camid = '_c' + camid + '_'
            shutil.copyfile(img_path, dir + 'gallery/' + pid + camid + imgid + '.jpg')

if __name__ == '__main__':

    pass
