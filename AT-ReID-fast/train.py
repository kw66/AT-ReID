import json
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
from loss.said import HEAD_SCENARIOS
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

TASK_GROUP_NAMES = ("dt", "nt", "ad")
TIME_GROUP_NAMES = ("st", "lt")


def configure_environment(args):
    requested_gpu = str(args.gpu).strip()
    if requested_gpu.lower() in {"env", "auto", "scheduler"}:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible_devices:
            raise RuntimeError("--gpu env requires CUDA_VISIBLE_DEVICES to be preset by the launcher/scheduler.")
        print(f"Using existing CUDA_VISIBLE_DEVICES={visible_devices}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = requested_gpu
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
    criterion_said = saidloss_hdw(
        cid2pid=dataset.cid2pid,
        said=args.said,
        hdw=args.hdw,
    ).to(device)
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


def _format_scenario_metrics(prefix, metrics):
    joined = " ".join(f"{name}:{metrics[name]:.4f}" for name in HEAD_SCENARIOS)
    return f"{prefix} {joined}"


def _format_route_metrics(prefix, metrics):
    joined = " ".join(f"{name}:{metrics[name]:.4f}" for name in HEAD_SCENARIOS)
    return f"{prefix} {joined}"


def _format_named_metrics(prefix, metrics):
    joined = " ".join(f"{name}:{value:.4f}" for name, value in metrics.items())
    return f"{prefix} {joined}"


def _tensor_metric_dict(names, values, *, integer=False):
    materialized = values.detach().cpu().tolist()
    if integer:
        return {name: int(materialized[index]) for index, name in enumerate(names)}
    return {name: float(materialized[index]) for index, name in enumerate(names)}


def _append_epoch_stats(checkpoint_path, payload):
    stats_path = os.path.join(checkpoint_path, "epoch_head_stats.jsonl")
    with open(stats_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_one_epoch(epoch, args, model, base_model, trainloader, optimizer, criterion_said, criterion_tri, scaler, amp_dtype, device, global_start):
    model.train()
    if args.moae and hasattr(base_model, "set_moae_epoch"):
        base_model.set_moae_epoch(epoch)
    collect_train_stats = bool(getattr(args, "train_stats", False))
    train_loss, data_time, batch_time, grad, acc, loss_1, loss_2 = [AverageMeter() for _ in range(7)]
    if collect_train_stats:
        num_scenarios = len(HEAD_SCENARIOS)
        scenario_loss_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_loss_weight = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_base_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_loss_contrib_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_weight_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_effective_weight_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_raw_score_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_smoothed_score_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_class_count_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        scenario_sample_count_total = torch.zeros(num_scenarios, device=device, dtype=torch.long)
        task_group_weight_sum = torch.zeros(len(TASK_GROUP_NAMES), device=device, dtype=torch.float32)
        time_group_weight_sum = torch.zeros(len(TIME_GROUP_NAMES), device=device, dtype=torch.float32)
        moae_token_count_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        moae_left_select_count_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        moae_left_prob_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        moae_right_prob_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        moae_chosen_weight_sum = torch.zeros(num_scenarios, device=device, dtype=torch.float32)
        moae_route_names = None
        stats_batch_count = 0
    else:
        moae_route_names = None
    weight_mode = "disabled" if not collect_train_stats else "none"
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
            if collect_train_stats:
                loss1, loss1_details = criterion_said(logits, pids, cids, mids, return_details=True)
            else:
                loss1 = criterion_said(logits, pids, cids, mids, return_details=False)
                loss1_details = None
            if epoch > args.warmup_loss:
                loss2 = compute_triplet_loss(criterion_tri, embeddings, pids, mids)
            else:
                loss2 = imgs.new_zeros(())
            loss = loss1 + loss2
        moae_batch_stats = base_model.consume_moae_route_stats() if (args.moae and collect_train_stats) else None
        if loss1_details is not None:
            weight_mode = loss1_details["weight_mode"]

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
        if args.moae and hasattr(base_model, "update_moae_balance"):
            base_model.update_moae_balance()

        predicted = logits[5].float().max(1)[1]
        train_loss.update(loss.item(), pids.size(0))
        loss_1.update(loss1.item(), pids.size(0))
        loss_2.update(loss2.item(), pids.size(0))
        grad.update(float(grad_norm))
        acc.update(100.0 * predicted.eq(pids).sum().item() / pids.size(0))
        if collect_train_stats and loss1_details is not None:
            scenario_counts = loss1_details["scenario_sample_counts"].to(device=device)
            scenario_counts_float = scenario_counts.to(dtype=torch.float32)
            scenario_sample_count_total += scenario_counts
            scenario_loss_sum += loss1_details["scenario_mean_losses"] * scenario_counts_float
            scenario_loss_weight += scenario_counts_float
            scenario_base_sum += loss1_details["scenario_base_losses"]
            scenario_loss_contrib_sum += loss1_details["scenario_loss_contribs"]
            scenario_weight_sum += loss1_details["scenario_weights"]
            scenario_effective_weight_sum += loss1_details["scenario_effective_weights"]
            if "scenario_raw_scores" in loss1_details:
                scenario_raw_score_sum += loss1_details["scenario_raw_scores"]
            if "scenario_smoothed_scores" in loss1_details:
                scenario_smoothed_score_sum += loss1_details["scenario_smoothed_scores"]
            if "scenario_class_count_means" in loss1_details:
                scenario_class_count_sum += loss1_details["scenario_class_count_means"]
            if "task_group_weights" in loss1_details:
                task_group_weight_sum += loss1_details["task_group_weights"]
            if "time_group_weights" in loss1_details:
                time_group_weight_sum += loss1_details["time_group_weights"]
            if moae_batch_stats is not None:
                moae_token_count_sum += moae_batch_stats["token_counts"].to(device=device)
                moae_left_select_count_sum += moae_batch_stats["left_select_counts"].to(device=device)
                moae_left_prob_sum += moae_batch_stats["left_prob_sums"].to(device=device)
                moae_right_prob_sum += moae_batch_stats["right_prob_sums"].to(device=device)
                moae_chosen_weight_sum += moae_batch_stats["chosen_weight_sums"].to(device=device)
                if moae_route_names is None:
                    moae_route_names = moae_batch_stats["route_names"]
            stats_batch_count += 1
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
    if not collect_train_stats:
        return {"weight_mode": weight_mode}
    batch_count = max(int(stats_batch_count), 1)
    zero_tensor = torch.zeros_like(scenario_loss_sum)
    scenario_mean_loss_tensor = torch.where(
        scenario_loss_weight > 0,
        scenario_loss_sum / scenario_loss_weight.clamp_min(1.0),
        zero_tensor,
    )
    scenario_base_loss_tensor = scenario_base_sum / batch_count
    scenario_loss_contrib_tensor = scenario_loss_contrib_sum / batch_count
    scenario_weight_tensor = scenario_weight_sum / batch_count
    scenario_effective_weight_tensor = scenario_effective_weight_sum / batch_count
    scenario_raw_score_tensor = scenario_raw_score_sum / batch_count
    scenario_smoothed_score_tensor = scenario_smoothed_score_sum / batch_count
    scenario_class_count_tensor = scenario_class_count_sum / batch_count
    task_group_weight_tensor = task_group_weight_sum / batch_count
    time_group_weight_tensor = time_group_weight_sum / batch_count
    moae_left_select_ratio_tensor = torch.where(
        moae_token_count_sum > 0,
        moae_left_select_count_sum / moae_token_count_sum.clamp_min(1.0),
        zero_tensor,
    )
    moae_right_select_ratio_tensor = torch.where(
        moae_token_count_sum > 0,
        (moae_token_count_sum - moae_left_select_count_sum) / moae_token_count_sum.clamp_min(1.0),
        zero_tensor,
    )
    moae_left_prob_mean_tensor = torch.where(
        moae_token_count_sum > 0,
        moae_left_prob_sum / moae_token_count_sum.clamp_min(1.0),
        zero_tensor,
    )
    moae_right_prob_mean_tensor = torch.where(
        moae_token_count_sum > 0,
        moae_right_prob_sum / moae_token_count_sum.clamp_min(1.0),
        zero_tensor,
    )
    moae_chosen_weight_mean_tensor = torch.where(
        moae_token_count_sum > 0,
        moae_chosen_weight_sum / moae_token_count_sum.clamp_min(1.0),
        zero_tensor,
    )
    epoch_stats = {
        "weight_mode": weight_mode,
        "scenario_mean_losses": _tensor_metric_dict(HEAD_SCENARIOS, scenario_mean_loss_tensor),
        "scenario_base_losses": _tensor_metric_dict(HEAD_SCENARIOS, scenario_base_loss_tensor),
        "scenario_loss_contribs": _tensor_metric_dict(HEAD_SCENARIOS, scenario_loss_contrib_tensor),
        "scenario_weights": _tensor_metric_dict(HEAD_SCENARIOS, scenario_weight_tensor),
        "scenario_effective_weights": _tensor_metric_dict(HEAD_SCENARIOS, scenario_effective_weight_tensor),
        "scenario_raw_scores": _tensor_metric_dict(HEAD_SCENARIOS, scenario_raw_score_tensor),
        "scenario_smoothed_scores": _tensor_metric_dict(HEAD_SCENARIOS, scenario_smoothed_score_tensor),
        "scenario_class_count_means": _tensor_metric_dict(HEAD_SCENARIOS, scenario_class_count_tensor),
        "scenario_sample_counts": _tensor_metric_dict(HEAD_SCENARIOS, scenario_sample_count_total, integer=True),
        "task_group_weights": _tensor_metric_dict(TASK_GROUP_NAMES, task_group_weight_tensor),
        "time_group_weights": _tensor_metric_dict(TIME_GROUP_NAMES, time_group_weight_tensor),
        "moae_left_select_ratio": _tensor_metric_dict(HEAD_SCENARIOS, moae_left_select_ratio_tensor),
        "moae_right_select_ratio": _tensor_metric_dict(HEAD_SCENARIOS, moae_right_select_ratio_tensor),
        "moae_left_prob_mean": _tensor_metric_dict(HEAD_SCENARIOS, moae_left_prob_mean_tensor),
        "moae_right_prob_mean": _tensor_metric_dict(HEAD_SCENARIOS, moae_right_prob_mean_tensor),
        "moae_chosen_weight_mean": _tensor_metric_dict(HEAD_SCENARIOS, moae_chosen_weight_mean_tensor),
        "moae_route_names": {
            scenario_name: {
                "left": route_names[0],
                "right": route_names[1],
            }
            for scenario_name, route_names in zip(HEAD_SCENARIOS, moae_route_names or ())
        },
    }
    print(_format_scenario_metrics("ScenarioLossMean", epoch_stats["scenario_mean_losses"]))
    print(_format_scenario_metrics("ScenarioBaseLossMean", epoch_stats["scenario_base_losses"]))
    print(_format_scenario_metrics("ScenarioLossContribMean", epoch_stats["scenario_loss_contribs"]))
    print(_format_scenario_metrics("ScenarioWeightMean", epoch_stats["scenario_weights"]))
    print(_format_scenario_metrics("ScenarioEffectiveWeightMean", epoch_stats["scenario_effective_weights"]))
    if any(value > 0 for value in epoch_stats["task_group_weights"].values()):
        print(_format_named_metrics("TaskGroupWeightMean", epoch_stats["task_group_weights"]))
    if any(value > 0 for value in epoch_stats["time_group_weights"].values()):
        print(_format_named_metrics("TimeGroupWeightMean", epoch_stats["time_group_weights"]))
    if any(value > 0 for value in epoch_stats["scenario_raw_scores"].values()):
        print(_format_scenario_metrics("ScenarioRawScoreMean", epoch_stats["scenario_raw_scores"]))
    if any(value > 0 for value in epoch_stats["scenario_smoothed_scores"].values()):
        print(_format_scenario_metrics("ScenarioSmoothedScoreMean", epoch_stats["scenario_smoothed_scores"]))
    if any(value > 0 for value in epoch_stats["scenario_class_count_means"].values()):
        print(_format_scenario_metrics("ScenarioClassCountMean", epoch_stats["scenario_class_count_means"]))
    if args.moae:
        print(_format_route_metrics("MOAELeftProbMean", epoch_stats["moae_left_prob_mean"]))
        print(_format_route_metrics("MOAERightProbMean", epoch_stats["moae_right_prob_mean"]))
        print(_format_route_metrics("MOAELeftSelectMean", epoch_stats["moae_left_select_ratio"]))
        print(_format_route_metrics("MOAEChosenWeightMean", epoch_stats["moae_chosen_weight_mean"]))
    return epoch_stats


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
        epoch_head_stats = train_one_epoch(epoch, args, model, base_model, trainloader, optimizer, criterion_said, criterion_tri, scaler, amp_dtype, device, global_start)
        if args.train_stats:
            _append_epoch_stats(
                checkpoint_path,
                {
                    "epoch": epoch,
                    "lr": float(current_lr),
                    "weight_mode": epoch_head_stats["weight_mode"],
                    "scenario_mean_losses": epoch_head_stats["scenario_mean_losses"],
                    "scenario_base_losses": epoch_head_stats["scenario_base_losses"],
                    "scenario_loss_contribs": epoch_head_stats["scenario_loss_contribs"],
                    "scenario_weights": epoch_head_stats["scenario_weights"],
                    "scenario_effective_weights": epoch_head_stats["scenario_effective_weights"],
                    "scenario_raw_scores": epoch_head_stats["scenario_raw_scores"],
                    "scenario_smoothed_scores": epoch_head_stats["scenario_smoothed_scores"],
                    "scenario_class_count_means": epoch_head_stats["scenario_class_count_means"],
                    "scenario_sample_counts": epoch_head_stats["scenario_sample_counts"],
                    "task_group_weights": epoch_head_stats["task_group_weights"],
                    "time_group_weights": epoch_head_stats["time_group_weights"],
                    "moae_left_select_ratio": epoch_head_stats["moae_left_select_ratio"],
                    "moae_right_select_ratio": epoch_head_stats["moae_right_select_ratio"],
                    "moae_left_prob_mean": epoch_head_stats["moae_left_prob_mean"],
                    "moae_right_prob_mean": epoch_head_stats["moae_right_prob_mean"],
                    "moae_chosen_weight_mean": epoch_head_stats["moae_chosen_weight_mean"],
                    "moae_route_names": epoch_head_stats["moae_route_names"],
                },
            )
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
