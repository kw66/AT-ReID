import os
import sys
from torch.backends import cudnn
import random
import torch
import numpy as np
import glob
import time
from tqdm import tqdm


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            if not os.path.exists(os.path.dirname(fpath)):
                os.mkdir(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)


def test_model(checkpoint_path, model):
    if os.path.isfile(os.path.join(checkpoint_path, 'complete.txt')):
        model_path = glob.glob(os.path.join(checkpoint_path, 'epoch_*.t'))
        if len(model_path) > 0:
            print(model_path[-1])
            checkpoint = torch.load(model_path[-1])
            model.load_state_dict(checkpoint['model'])
            return True
    return False


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def gpu_avaliable(args):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #'''
    if args.t > 0:
        for i in tqdm(range(int(args.t*3600))):
            time.sleep(1)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    while args.wait:
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
        gpu_memory = int(gpu_status[2 + 4 * int(args.gpu)].split('/')[0].split('M')[0].strip())
        if gpu_memory < args.mb:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()
        else:
            time.sleep(20)
    #'''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()


def mkdir_(args):
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    while os.path.exists(f'{args.log_path}/{args.d}_v{args.v}.txt'):
        if args.test:
            break
        if os.path.isfile(f'{args.model_path}/{args.d}_v{args.v}/complete.txt'):
            args.v += 1
        else:
            os.remove(f'{args.log_path}/{args.d}_v{args.v}.txt')
    checkpoint_path = f'{args.model_path}/{args.d}_v{args.v}/'
    log_path = f'{args.log_path}/{args.d}_v{args.v}.txt'
    if not args.test:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        sys.stdout = Logger(log_path)
    return checkpoint_path
