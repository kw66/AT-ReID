import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn


class Logger:
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

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

    def isatty(self):
        if hasattr(self.console, "isatty"):
            return self.console.isatty()
        return False

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.close()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def set_seed(seed, deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters:
        return 0.0
    norm_type = float(norm_type)
    total_norm = 0.0
    for param in parameters:
        param_norm = param.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def start_timer():
    print(now_str())
    return now_str(), time.time()


def make_output_dirs(args):
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    while os.path.exists(f'{args.log_path}/{args.dataset}_v{args.v}.txt'):
        if args.test or args.eval_only or args.resume or args.checkpoint:
            break
        if os.path.isfile(f'{args.model_path}/{args.dataset}_v{args.v}/complete.txt'):
            args.v += 1
        else:
            os.remove(f'{args.log_path}/{args.dataset}_v{args.v}.txt')
    checkpoint_path = f'{args.model_path}/{args.dataset}_v{args.v}/'
    log_path = f'{args.log_path}/{args.dataset}_v{args.v}.txt'
    if not args.test and not args.eval_only:
        os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = Logger(log_path)
    return checkpoint_path, log_path


def find_checkpoint(checkpoint_path, explicit_checkpoint=None):
    if explicit_checkpoint:
        return explicit_checkpoint
    best_path = os.path.join(checkpoint_path, 'epoch_best.t')
    if os.path.isfile(best_path):
        return best_path
    model_paths = sorted(glob.glob(os.path.join(checkpoint_path, 'epoch_*.t')))
    if model_paths:
        return model_paths[-1]
    return None


def load_checkpoint(model, checkpoint_path, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def maybe_load_existing_checkpoint(checkpoint_path, model, explicit_checkpoint=None):
    candidate = find_checkpoint(checkpoint_path, explicit_checkpoint=explicit_checkpoint)
    if candidate is None:
        return False, None
    print(candidate)
    checkpoint = load_checkpoint(model, candidate)
    return True, checkpoint


def save_checkpoint(model, checkpoint_path, epoch, is_best=True, save_every=0):
    state = {'model': model.state_dict(), 'epoch': epoch}
    if is_best:
        torch.save(state, os.path.join(checkpoint_path, 'epoch_best.t'))
    if save_every > 0 and epoch % save_every == 0:
        torch.save(state, os.path.join(checkpoint_path, f'epoch_{epoch:03d}.t'))


def write_complete_log(checkpoint_path, log_text):
    with open(os.path.join(checkpoint_path, 'complete.txt'), 'w', encoding='utf-8') as f:
        f.write(log_text)


def write_summary_log(dataset_name, log_text, log_root=None):
    if log_root is None:
        summary_path = f'./{dataset_name}log.txt'
    else:
        os.makedirs(log_root, exist_ok=True)
        summary_path = os.path.join(log_root, f'{dataset_name}_summary.txt')
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(log_text)


def format_hms(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600:d}h{seconds // 60 % 60:d}m{seconds % 60:d}s'


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
