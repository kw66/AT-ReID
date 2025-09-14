import os
import glob
import re
import shutil
from tqdm import tqdm

dir = "/home/lixulin/data/prcc/"
if not os.path.isdir(dir+'train/'):
    os.makedirs(dir+'train/')
if not os.path.isdir(dir+'query/'):
    os.makedirs(dir+'query/')
if not os.path.isdir(dir+'gallery/'):
    os.makedirs(dir+'gallery/')

img_paths = glob.glob(dir + 'rgb/train/*/*.[jp][pn]g') + glob.glob(dir + 'rgb/val/*/*.[jp][pn]g')
pattern = re.compile(r'(\d+)/([ABC])_cropped_rgb(\d+)')
for img_path in tqdm(img_paths):
    pid, camid, imgid = pattern.search(img_path).groups()
    if camid == 'A':
        camid_cid = '_c1_1_'
    elif camid == 'B':
        camid_cid = '_c2_1_'
    elif camid == 'C':
        camid_cid = '_c3_2_'
    #img = Image.open(img_path).convert('RGB')
    shutil.copyfile(img_path, dir + 'train/' + pid + camid_cid + imgid + '.jpg')

img_paths = glob.glob(dir + 'rgb/test/*/*/*.[jp][pn]g')
pattern = re.compile(r'([ABC])/(\d+)/cropped_rgb(\d+)')
for img_path in tqdm(img_paths):
    camid, pid, imgid = pattern.search(img_path).groups()
    if camid == 'A':
        camid_cid = '_c1_1_'
        shutil.copyfile(img_path, dir + 'gallery/' + pid + '_c1_1_' + imgid + '.jpg')
    elif camid == 'B':
        camid_cid = '_c2_1_'
        shutil.copyfile(img_path, dir + 'query/' + pid + camid_cid + imgid + '.jpg')
    elif camid == 'C':
        camid_cid = '_c3_2_'
        shutil.copyfile(img_path, dir + 'query/' + pid + camid_cid + imgid + '.jpg')

if __name__ == '__main__':
    pass
