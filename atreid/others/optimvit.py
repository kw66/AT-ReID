import copy
import torch
import math
from torch import nn


def get_optim(args, param_groups):
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    if args.optim == 'adamW':
        optimizer = torch.optim.AdamW(param_groups)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    return optimizer


def get_param_groups(args, lr, model):
    memo = set()
    param_groups = []
    defaults = {}
    defaults["lr"] = lr
    defaults["weight_decay"] = args.wd
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            hyperparams = copy.copy(defaults)
            if "base" in module_name:
                scale = 1
                if "blocks" in module_name:
                    layer_id = int(module_name.split('.')[2])
                elif "base.norm" in module_name:
                    layer_id = 12
                else:
                    layer_id = 0
                if layer_id in args.dlrl:
                    scale = args.dlr
                hyperparams["lr"] = hyperparams["lr"] * scale
            if "bottleneck" in module_name or "classifier" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * args.dlr
            if args.dwd:
              if "patch_embed" in module_name:
                  hyperparams["weight_decay"] = hyperparams["weight_decay"] * 1.0
              if "pos_embed" in module_param_name:
                  hyperparams["weight_decay"] = hyperparams["weight_decay"] * 0.0
              if "cls_token" in module_param_name:
                  hyperparams["weight_decay"] = hyperparams["weight_decay"] * 0.0
              if 'bias' in module_param_name:
                  hyperparams["weight_decay"] = hyperparams["weight_decay"] * 1.0
              if isinstance(module, nn.LayerNorm):
                  hyperparams["weight_decay"] = hyperparams["weight_decay"] * 0.0
            hyperparams["module_name"] = module_name
            param_groups.append({"params": [value], **hyperparams})
    return param_groups


def adjust_learning_rate(args, optimizer, epoch, model):
    lr = args.lr
    if epoch <= args.warmup_epoch:
        lr *= ((1 - args.warmup_rate) * (epoch - 1) / (args.warmup_epoch - 1) + args.warmup_rate)
    elif len(args.scheduler_epoch) > 0:
        for scheduler_epoch in args.scheduler_epoch:
            if epoch > scheduler_epoch:
                lr *= args.scheduler_rate
    else:
        step_ratio = (epoch - args.warmup_epoch) / (args.max_epoch - args.warmup_epoch)
        lr *= ((1 - args.cosmin_rate) * (1 + math.cos(math.pi * step_ratio)) / 2 + args.cosmin_rate)
    new_lr = [param_group['lr'] for param_group in get_param_groups(args, lr, model)]
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = new_lr[i]
    return lr


