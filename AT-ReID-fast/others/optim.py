import copy
import inspect
import math

import torch
from torch import nn


def get_optim(args, param_groups):
    optim_name = args.optim.lower()
    fused = False
    if args.optimizer_fused and optim_name in {"adam", "adamw"} and torch.cuda.is_available():
        optimizer_cls = torch.optim.Adam if optim_name == "adam" else torch.optim.AdamW
        if "fused" in inspect.signature(optimizer_cls).parameters:
            fused = True
        else:
            print("optimizer_fused was requested, but this torch build does not expose fused Adam/AdamW. Falling back.")

    if optim_name == "adam":
        return torch.optim.Adam(param_groups, fused=fused)
    if optim_name == "adamw":
        return torch.optim.AdamW(param_groups, fused=fused)
    if args.optimizer_fused:
        print("optimizer_fused is only applicable to Adam/AdamW. Falling back to standard SGD.")
    return torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)


def get_param_groups(args, lr, model):
    if lr == 0 and getattr(args, "group_optimizer", True):
        # Preserve the multiplier distinction even when the scheduled LR is zero.
        groups = get_param_groups(args, 1.0, model)
        for group in groups:
            group["lr"] = 0.0
        return groups
    memo = set()
    param_groups = []
    defaults = {"lr": lr, "weight_decay": args.wd}
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
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
                hyperparams["lr"] *= scale
            if "bottleneck" in module_name or "classifier" in module_name:
                hyperparams["lr"] *= args.dlr
            if args.dwd:
                if "pos_embed" in module_param_name or "cls_token" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, nn.LayerNorm):
                    hyperparams["weight_decay"] = 0.0
            hyperparams["module_name"] = module_name
            if getattr(args, "group_optimizer", True):
                hyperparams["parameter_name"] = f"{module_name}.{module_param_name}".lstrip(".")
            param_groups.append({"params": [value], **hyperparams})
    if not getattr(args, "group_optimizer", True):
        return param_groups
    grouped = {}
    for group in param_groups:
        options = {key: value for key, value in group.items() if key not in {"params", "module_name", "parameter_name"}}
        key = tuple(sorted(options.items()))
        if key not in grouped:
            grouped[key] = {**options, "params": [], "param_names": [], "lr_scale": options["lr"] / lr}
        grouped[key]["params"].extend(group["params"])
        grouped[key]["param_names"].append(group["parameter_name"])
    return list(grouped.values())


def adjust_learning_rate(args, optimizer, epoch, model):
    lr = args.lr
    if epoch <= args.warmup_epoch:
        lr *= ((1 - args.warmup_rate) * (epoch - 1) / max(1, args.warmup_epoch - 1) + args.warmup_rate)
    elif len(args.scheduler_epoch) > 0:
        for scheduler_epoch in args.scheduler_epoch:
            if epoch > scheduler_epoch:
                lr *= args.scheduler_rate
    else:
        step_ratio = (epoch - args.warmup_epoch) / max(1, (args.max_epoch - args.warmup_epoch))
        lr *= ((1 - args.cosmin_rate) * (1 + math.cos(math.pi * step_ratio)) / 2 + args.cosmin_rate)
    if all("lr_scale" in group for group in optimizer.param_groups):
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        return lr
    new_lr = [param_group['lr'] for param_group in get_param_groups(args, lr, model)]
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr[i]
    return lr
