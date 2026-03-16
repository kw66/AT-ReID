import os
import time
from types import SimpleNamespace

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from args.args import parse_args
from dataset.atustc import atustc
from dataset.datamanager import ImageDataset
from dataset.dataset_all import dataset_all
from dataset.sample import PSKSampler
from loss.hdw import saidloss_hdw
from loss.tri import TripletLoss
from model.uniat import uniat
from others.optim import adjust_learning_rate, get_optim, get_param_groups
from others.runtime import (
    autocast_context,
    build_dataloader_kwargs,
    create_grad_scaler,
    format_vit_attention_info,
    get_device,
    maybe_compile_model,
    resolve_distance_device,
    select_amp_dtype,
    setup_runtime,
)
from others.test import attest
from others.test_all import test_all
from others.transforms import get_transform
from others.utils import (
    AverageMeter,
    find_checkpoint,
    format_hms,
    get_grad_norm,
    load_checkpoint,
    make_output_dirs,
    now_str,
    save_checkpoint,
    set_seed,
    start_timer,
    write_complete_log,
    write_summary_log,
)


def configure_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    setup_runtime(args)
    set_seed(args.seed, deterministic=args.deterministic)


def build_primary_dataset(args):
    return atustc(data_dir=args.data_root, data_root_config=args.data_root_config)


def build_trainloader(args, dataset):
    sampler = PSKSampler(dataset.train, args.p, args.k, args.n, args.sample)
    transform_train, _ = get_transform(args)
    kwargs = build_dataloader_kwargs(args, train=True)
    return DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sampler,
        batch_size=args.p * args.k,
        **kwargs,
    )


def build_model(args, dataset, device):
    base_model = uniat(
        num_p=dataset.num_p,
        num_c=dataset.num_c,
        imsize=(args.ih, args.iw),
        drop=args.drop,
        stride=args.stride,
        ncls=args.ncls,
        moae=args.moae,
        moae_router_noise=args.moae_router_noise,
        use_pretrained=not args.no_pretrained,
        pretrained_path=args.pretrained_path,
        attention_backend=args.vit_attention_backend,
    ).to(device)
    compile_args = args
    if args.test or args.eval_only:
        compile_args = SimpleNamespace(compile=False, compile_mode=getattr(args, "compile_mode", "default"))
    train_model = maybe_compile_model(base_model, compile_args)
    return base_model, train_model


def build_losses(args, dataset, device):
    criterion_said = saidloss_hdw(cid2pid=dataset.cid2pid, said=args.said, hdw=args.hdw).to(device)
    criterion_tri = TripletLoss().to(device)
    return criterion_said, criterion_tri


def _triplet_if_valid(criterion_tri, features, pids):
    if features.shape[0] < 2:
        return features.new_zeros(())
    if torch.unique(pids).numel() < 2:
        return features.new_zeros(())
    return criterion_tri(features, pids)


def compute_triplet_loss(criterion_tri, embeddings, pids, mids):
    loss = embeddings[0].new_zeros(())
    rgb_mask = mids == 1
    ir_mask = mids == 2
    if rgb_mask.any():
        loss = loss + _triplet_if_valid(criterion_tri, embeddings[0][rgb_mask], pids[rgb_mask])
        loss = loss + _triplet_if_valid(criterion_tri, embeddings[1][rgb_mask], pids[rgb_mask])
    if ir_mask.any():
        loss = loss + _triplet_if_valid(criterion_tri, embeddings[2][ir_mask], pids[ir_mask])
        loss = loss + _triplet_if_valid(criterion_tri, embeddings[3][ir_mask], pids[ir_mask])
    loss = loss + _triplet_if_valid(criterion_tri, embeddings[4], pids)
    loss = loss + _triplet_if_valid(criterion_tri, embeddings[5], pids)
    return loss


def train_one_epoch(epoch, args, model, base_model, trainloader, optimizer, criterion_said, criterion_tri, scaler, amp_dtype, device, global_start):
    model.train()
    train_loss, data_time, batch_time, grad, acc, loss_1, loss_2 = [AverageMeter() for _ in range(7)]
    end = time.time()
    for batch_idx, (imgs, pids, cids, mids, camids, _) in enumerate(trainloader):
        optimizer.zero_grad(set_to_none=args.zero_grad_set_to_none)
        imgs = imgs.to(device, non_blocking=args.non_blocking)
        pids = pids.to(device, non_blocking=args.non_blocking)
        cids = cids.to(device, non_blocking=args.non_blocking)
        mids = mids.to(device, non_blocking=args.non_blocking)
        camids = camids.to(device, non_blocking=args.non_blocking)
        data_time.update(time.time() - end)

        with autocast_context(args.amp, amp_dtype, device.type):
            embeddings, logits = model(imgs)
            loss1 = criterion_said(logits, pids, cids, mids)
            if epoch > args.warmup_loss:
                loss2 = compute_triplet_loss(criterion_tri, embeddings, pids, mids)
            else:
                loss2 = imgs.new_zeros(())
            loss = loss1 + loss2

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip > 0:
                grad_norm = clip_grad_norm_(base_model.parameters(), args.clip)
            else:
                grad_norm = get_grad_norm(base_model.parameters())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip > 0:
                grad_norm = clip_grad_norm_(base_model.parameters(), args.clip)
            else:
                grad_norm = get_grad_norm(base_model.parameters())
            optimizer.step()

        predicted = logits[5].float().max(1)[1]
        train_loss.update(loss.item(), pids.size(0))
        loss_1.update(loss1.item(), pids.size(0))
        loss_2.update(loss2.item(), pids.size(0))
        grad.update(float(grad_norm))
        acc.update(100.0 * predicted.eq(pids).sum().item() / pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        step = epoch - 1 + (batch_idx + 1) / len(trainloader)
        elapsed = time.time() - global_start
        remaining = elapsed * (args.max_epoch - step) / max(step, 1e-8)
        if (batch_idx + 1) % max(1, len(trainloader) // 10) == 0:
            print(
                f'E[{epoch:02d}][{batch_idx + 1:3d}/{len(trainloader)}] '
                f'L: {train_loss.val:.3f} ({train_loss.avg:.3f}) '
                f'L1: {loss_1.val:.3f} ({loss_1.avg:.3f}) '
                f'L2: {loss_2.val:.3f} ({loss_2.avg:.3f}) '
                f'Acc: {acc.val:<4.1f} ({acc.avg:<4.1f}) '
                f'Grad: {grad.val:<4.2f} ({grad.avg:<4.2f}) '
                f'Time: {format_hms(elapsed)}/{format_hms(remaining)}'
            )
        if args.max_train_steps > 0 and (batch_idx + 1) >= args.max_train_steps:
            print(f'==> Early stop epoch {epoch} after {batch_idx + 1} training steps (max_train_steps={args.max_train_steps})')
            break
    print(f'DataTime: {data_time.sum:.3f} ({data_time.avg:.3f}) BatchTime: {batch_time.sum:.3f} ({batch_time.avg:.3f})')


def evaluate_model(args, model, dataset):
    cmc, mAP = attest(args, dataset, model)
    if args.test_all:
        for dataset_name in ['market', 'cuhk', 'msmt', 'sysu', 'regdb', 'llcm', 'prcc', 'ltcc', 'deepchange']:
            print(f'==> Cross-dataset test: {dataset_name}')
            test_all(
                args,
                dataset_name,
                dataset_all(dataset_name, data_root_config=args.data_root_config),
                model,
            )
    return cmc, mAP


def run(args):
    configure_environment(args)
    start_time_str, global_start = start_timer()
    checkpoint_path, _ = make_output_dirs(args)
    device = get_device()
    dataset = build_primary_dataset(args)
    base_model, model = build_model(args, dataset, device)
    print(f'Runtime Mode: {args.runtime_mode} | AMP: {args.amp} ({args.amp_dtype}) | Compile(train): {args.compile}')
    print(f'ViT Attention: {format_vit_attention_info(getattr(base_model, "vit_attention_info", None))}')
    distance_device = resolve_distance_device(args.test_distance_device, default_device=device)
    print(
        f'Test Distance: device={distance_device.type} '
        f'rank={getattr(args, "test_rank_device", "auto")} '
        f'chunk={args.test_distance_chunk_size} '
        f'decode_cache={getattr(args, "decode_cache", "off")} '
        f'test_batch={args.test_batch} '
        f'auto_batch={bool(getattr(args, "test_batch_auto_tune", False))} '
        f'test_amp={args.test_amp} '
        f'inference_mode={args.inference_mode}'
    )

    if args.resume:
        checkpoint = load_checkpoint(base_model, args.resume, map_location=device)
        print(f"Resumed model weights from {args.resume} (epoch {checkpoint.get('epoch', 'unknown')})")

    if args.test or args.eval_only:
        checkpoint_file = find_checkpoint(checkpoint_path, explicit_checkpoint=args.checkpoint or args.resume)
        if checkpoint_file is None:
            raise FileNotFoundError("No checkpoint found. Please pass --checkpoint or train a model first.")
        load_checkpoint(base_model, checkpoint_file, map_location=device)
        print(f'Loaded checkpoint for evaluation: {checkpoint_file}')
        return evaluate_model(args, base_model, dataset)

    trainloader = build_trainloader(args, dataset)
    param_groups = get_param_groups(args, args.lr, base_model)
    optimizer = get_optim(args, param_groups)
    criterion_said, criterion_tri = build_losses(args, dataset, device)
    amp_dtype = select_amp_dtype(args.amp_dtype)
    scaler = create_grad_scaler(args.amp, amp_dtype)

    print('==> Start Training...')
    best_acc, best_epoch = 0.0, 0
    final_cmc, final_mAP = None, None
    for epoch in range(1, args.max_epoch + 1):
        current_lr = adjust_learning_rate(args, optimizer, epoch, base_model)
        print(
            f'==> Start Training Epoch: {epoch}   '
            f'lr {current_lr:.6f}   {args.dataset} v{args.v}   gpu {args.gpu}   best {best_acc:.2%}({best_epoch})'
        )
        train_one_epoch(epoch, args, model, base_model, trainloader, optimizer, criterion_said, criterion_tri, scaler, amp_dtype, device, global_start)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(base_model, checkpoint_path, epoch, is_best=False, save_every=args.save_every)
        if args.skip_eval:
            continue
        if epoch % args.test_epoch == 0 or epoch >= args.max_epoch - args.last_test:
            print(f'==> Start Testing Epoch: {epoch}')
            cmc, mAP = evaluate_model(args, base_model, dataset)
            final_cmc, final_mAP = cmc, mAP
            if cmc[0] > best_acc:
                best_acc, best_epoch = float(cmc[0]), epoch
                save_checkpoint(base_model, checkpoint_path, epoch, is_best=True, save_every=args.save_every)

    if args.skip_eval:
        log = f'{args.v:-3d}  Training finished without evaluation (skip_eval=True)\n'
        print(log.strip())
        write_summary_log(args.dataset, log, args.log_path)
        write_complete_log(checkpoint_path, log)
        return None, None
    if final_cmc is None or final_mAP is None:
        print('==> Final evaluation (no scheduled test was triggered during training).')
        final_cmc, final_mAP = evaluate_model(args, base_model, dataset)
        if final_cmc[0] > best_acc:
            best_acc, best_epoch = float(final_cmc[0]), args.max_epoch
            save_checkpoint(base_model, checkpoint_path, args.max_epoch, is_best=True, save_every=args.save_every)

    print(f'{final_cmc[0] * 100:-2.2f}\t{final_mAP * 100:-2.2f}')
    print(f'start:{start_time_str}\n end :{now_str()}')
    log = (
        f'{args.v:-3d}  Rank-1: {final_cmc[0]:.2%}  Rank-5: {final_cmc[4]:.2%}  Rank-10: {final_cmc[9]:.2%}  '
        f'Rank-20: {final_cmc[19]:.2%}  mAP: {final_mAP:.2%} \n'
    )
    write_summary_log(args.dataset, log, args.log_path)
    write_complete_log(checkpoint_path, log)
    return final_cmc, final_mAP


def main():
    parsed_args = parse_args()
    run(parsed_args)


if __name__ == '__main__':
    main()
