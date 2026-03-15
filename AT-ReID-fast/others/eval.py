import numpy as np
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.datamanager import ImageDataset
from others.runtime import autocast_context, build_dataloader_kwargs, resolve_distance_device, resolve_rank_device


FEATURE_PAIR_TO_HEAD = {
    ("vm", "sc"): 0,
    ("vm", "cc"): 1,
    ("im", "sc"): 2,
    ("im", "cc"): 3,
    ("cm", "sc"): 4,
    ("cm", "cc"): 5,
}


def _grad_free_context(use_inference_mode: bool):
    return torch.inference_mode() if use_inference_mode else torch.no_grad()


def _as_feature_list(model_output):
    if isinstance(model_output, (list, tuple)):
        return list(model_output)
    return [model_output]


def _empty_feature_tensor(num_rows: int = 0):
    return torch.empty((num_rows, 0), dtype=torch.float32)


def _extract_feature_heads_from_loader(
    model,
    dataloader,
    *,
    flip=False,
    use_amp=False,
    amp_dtype=None,
    use_inference_mode=True,
    non_blocking=True,
    device=None,
):
    model.eval()
    device = device or next(model.parameters()).device
    feature_lists = None
    pid_s, cid_s, mid_s, camid_s = [], [], [], []
    with _grad_free_context(use_inference_mode):
        for imgs, pids, cids, mids, camids, _ in dataloader:
            pid_s.extend(pids)
            cid_s.extend(cids)
            mid_s.extend(mids)
            camid_s.extend(camids)
            imgs = imgs.to(device, non_blocking=non_blocking)
            with autocast_context(use_amp, amp_dtype, device.type):
                feats = _as_feature_list(model(imgs))
                if flip:
                    flipped_feats = _as_feature_list(model(torch.flip(imgs, dims=[3])))
                    if len(flipped_feats) != len(feats):
                        raise RuntimeError(
                            "Model returned a different number of feature heads for the flipped branch."
                        )
                    feats = [feat + feat_flip for feat, feat_flip in zip(feats, flipped_feats)]
            if feature_lists is None:
                feature_lists = [[] for _ in range(len(feats))]
            for head_idx, feat in enumerate(feats):
                feature_lists[head_idx].append(feat.float().cpu())
    if feature_lists is None:
        stacked = [_empty_feature_tensor()]
    else:
        stacked = [torch.cat(chunks, dim=0) if chunks else _empty_feature_tensor() for chunks in feature_lists]
    return (
        stacked,
        np.asarray(pid_s, dtype=np.int64),
        np.asarray(cid_s, dtype=np.int64),
        np.asarray(mid_s, dtype=np.int64),
        np.asarray(camid_s, dtype=np.int64),
    )


def extract_feature(
    model,
    dataloader,
    flip=False,
    use_amp=False,
    amp_dtype=None,
    use_inference_mode=True,
    non_blocking=True,
    device=None,
):
    heads, pid_s, cid_s, mid_s, camid_s = _extract_feature_heads_from_loader(
        model,
        dataloader,
        flip=flip,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_inference_mode=use_inference_mode,
        non_blocking=non_blocking,
        device=device,
    )
    feat_s = heads[0] if heads else _empty_feature_tensor()
    return feat_s, pid_s, cid_s, mid_s, camid_s


def extract_feature6(
    model,
    dataloader,
    flip=False,
    use_amp=False,
    amp_dtype=None,
    use_inference_mode=True,
    non_blocking=True,
    device=None,
):
    heads, pid_s, cid_s, mid_s, camid_s = _extract_feature_heads_from_loader(
        model,
        dataloader,
        flip=flip,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_inference_mode=use_inference_mode,
        non_blocking=non_blocking,
        device=device,
    )
    if len(heads) < 6:
        if len(heads) == 1:
            heads = heads * 6
        else:
            raise RuntimeError(f"Expected 6 feature heads, but got {len(heads)}.")
    return (*heads[:6], pid_s, cid_s, mid_s, camid_s)


def _normalize_samples(samples):
    if isinstance(samples, np.ndarray):
        return [tuple(sample) for sample in samples.tolist()]
    return [tuple(sample) for sample in samples]


def _sample_feature_key(sample, model, args, device, *, amp_dtype):
    return (
        id(model),
        str(sample[0]),
        int(getattr(args, "ih", 256)),
        int(getattr(args, "iw", 128)),
        bool(getattr(args, "flip", False)),
        bool(getattr(args, "test_amp", False)),
        str(amp_dtype),
        str(device.type),
        str(getattr(args, "decode_cache", "off")),
    )


def _bundle_cache_key(sample_keys):
    return tuple(sample_keys)


def _feature_cache_namespace(feature_cache: dict | None):
    if feature_cache is None:
        return None
    namespace = feature_cache.get("__sample_feature_cache__")
    if namespace is None:
        namespace = {}
        feature_cache["__sample_feature_cache__"] = namespace
    return namespace


def _bundle_cache_namespace(feature_cache: dict | None):
    if feature_cache is None:
        return None
    namespace = feature_cache.get("__sample_bundle_cache__")
    if namespace is None:
        namespace = {}
        feature_cache["__sample_bundle_cache__"] = namespace
    return namespace


def _test_batch_cache_namespace(feature_cache: dict | None):
    if feature_cache is None:
        return None
    namespace = feature_cache.get("__test_batch_cache__")
    if namespace is None:
        namespace = {}
        feature_cache["__test_batch_cache__"] = namespace
    return namespace


def _stack_head_features(heads):
    if len(heads) == 1:
        return heads[0].unsqueeze(1)
    return torch.stack(heads, dim=1)


def _bundle_from_sample_cache(normalized_samples, sample_keys, sample_feature_cache):
    if sample_keys:
        feat = torch.stack([sample_feature_cache[sample_key] for sample_key in sample_keys], dim=0)
    else:
        feat = torch.empty((0, 1, 0), dtype=torch.float32)
    return {
        "feat": feat,
        "pids": np.asarray([int(sample[1]) for sample in normalized_samples], dtype=np.int64),
        "cids": np.asarray([int(sample[2]) for sample in normalized_samples], dtype=np.int64),
        "mids": np.asarray([int(sample[3]) for sample in normalized_samples], dtype=np.int64),
        "camids": np.asarray([int(sample[4]) for sample in normalized_samples], dtype=np.int64),
    }


def _decode_cache_mode(args):
    return str(getattr(args, "decode_cache", "off")).strip().lower()


def _build_eval_loader(
    samples,
    transform,
    args,
    *,
    batch_size,
    decode_cache_mode,
    decode_cache_dir,
    workers=None,
    verbose=False,
):
    kwargs = build_dataloader_kwargs(args, train=False)
    if workers is not None:
        kwargs["num_workers"] = int(workers)
        if int(workers) <= 0:
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)
        elif args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        ImageDataset(
            samples,
            transform=transform,
            decode_cache=decode_cache_mode,
            cache_resize=(args.ih, args.iw) if decode_cache_mode in {"ram", "disk"} else None,
            cache_dir=decode_cache_dir,
            verbose=verbose,
        ),
        batch_size=batch_size,
        **kwargs,
    )


def _test_batch_cache_key(model, args, device, *, amp_dtype, decode_cache_mode):
    return (
        id(model),
        int(getattr(args, "ih", 256)),
        int(getattr(args, "iw", 128)),
        bool(getattr(args, "flip", False)),
        bool(getattr(args, "test_amp", False)),
        bool(getattr(args, "inference_mode", True)),
        str(amp_dtype),
        str(device.type),
        str(decode_cache_mode),
    )


def _is_oom_error(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _probe_eval_batch_size(
    model,
    samples,
    transform,
    args,
    *,
    amp_dtype,
    device,
    batch_size,
    decode_cache_mode,
    decode_cache_dir,
):
    if batch_size <= 0 or not samples:
        return 0.0
    probe_samples = samples[:batch_size]
    start = time.perf_counter()
    loader = _build_eval_loader(
        probe_samples,
        transform,
        args,
        batch_size=batch_size,
        decode_cache_mode=decode_cache_mode,
        decode_cache_dir=decode_cache_dir,
        workers=0,
        verbose=False,
    )
    _extract_feature_heads_from_loader(
        model,
        loader,
        flip=args.flip,
        use_amp=args.test_amp,
        amp_dtype=amp_dtype,
        use_inference_mode=args.inference_mode,
        non_blocking=args.non_blocking,
        device=device,
    )
    return time.perf_counter() - start


def _resolve_eval_batch_size(
    model,
    samples,
    transform,
    args,
    *,
    amp_dtype,
    device,
    decode_cache_mode,
    decode_cache_dir,
    feature_cache,
):
    if not samples:
        return max(1, int(getattr(args, "test_batch", 1)))

    base_batch = max(1, min(int(getattr(args, "test_batch", 1)), len(samples)))
    if not bool(getattr(args, "test_batch_auto_tune", False)) or device.type != "cuda" or not torch.cuda.is_available():
        return base_batch

    cache = _test_batch_cache_namespace(feature_cache)
    cache_key = _test_batch_cache_key(
        model,
        args,
        device,
        amp_dtype=amp_dtype,
        decode_cache_mode=decode_cache_mode,
    )
    if cache is not None and cache_key in cache:
        return max(1, min(int(cache[cache_key]), len(samples)))

    max_auto = int(getattr(args, "test_batch_auto_max", 0) or 0)
    upper = len(samples) if max_auto <= 0 else min(len(samples), max(base_batch, max_auto))

    def _try_batch(batch_size):
        try:
            elapsed = _probe_eval_batch_size(
                model,
                samples,
                transform,
                args,
                amp_dtype=amp_dtype,
                device=device,
                batch_size=batch_size,
                decode_cache_mode=decode_cache_mode,
                decode_cache_dir=decode_cache_dir,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True, float(elapsed)
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, None

    seed_batch = base_batch
    ok, elapsed = _try_batch(seed_batch)
    while not ok and seed_batch > 1:
        seed_batch = max(1, seed_batch // 2)
        ok, elapsed = _try_batch(seed_batch)
    if not ok:
        return 1

    candidates = []
    current = seed_batch
    current_elapsed = float(elapsed)
    candidates.append((current, current_elapsed))
    while current < upper:
        next_batch = min(upper, current * 2)
        if next_batch == current:
            break
        ok, elapsed = _try_batch(next_batch)
        if not ok:
            break
        current = next_batch
        current_elapsed = float(elapsed)
        candidates.append((current, current_elapsed))

    best = min(candidates, key=lambda item: item[1] / max(1, item[0]))[0]

    if cache is not None:
        cache[cache_key] = int(best)
        feature_cache["__test_batch_cache__"] = cache

    if int(best) != int(base_batch):
        tried = ", ".join(str(batch) for batch, _ in candidates)
        print(f"Auto-tuned test_batch: {best} (base {base_batch}, max {upper}, tried [{tried}])")
    return int(best)


def _extract_missing_sample_features(
    model,
    missing_samples,
    transform,
    args,
    *,
    amp_dtype,
    device,
    sample_feature_cache,
    sample_keys,
    sample_indices,
    decode_cache_mode,
    decode_cache_dir,
    feature_cache,
):
    batch_size = _resolve_eval_batch_size(
        model,
        missing_samples,
        transform,
        args,
        amp_dtype=amp_dtype,
        device=device,
        decode_cache_mode=decode_cache_mode,
        decode_cache_dir=decode_cache_dir,
        feature_cache=feature_cache,
    )
    while True:
        try:
            loader = _build_eval_loader(
                missing_samples,
                transform,
                args,
                batch_size=batch_size,
                decode_cache_mode=decode_cache_mode,
                decode_cache_dir=decode_cache_dir,
                verbose=decode_cache_mode in {"ram", "disk"},
            )
            heads, _pid_s, _cid_s, _mid_s, _camid_s = _extract_feature_heads_from_loader(
                model,
                loader,
                flip=args.flip,
                use_amp=args.test_amp,
                amp_dtype=amp_dtype,
                use_inference_mode=args.inference_mode,
                non_blocking=args.non_blocking,
                device=device,
            )
            stacked_heads = _stack_head_features(heads)
            for local_idx, sample_idx in enumerate(sample_indices):
                sample_feature_cache[sample_keys[sample_idx]] = stacked_heads[local_idx].clone()
            return
        except RuntimeError as exc:
            if not _is_oom_error(exc) or batch_size <= 1:
                raise
            next_batch = max(1, batch_size // 2)
            print(f"Retry test feature extraction with smaller batch: {batch_size} -> {next_batch}")
            batch_size = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def extract_sample_feature_bundles(
    model,
    sample_groups,
    transform,
    args,
    *,
    amp_dtype=None,
    device=None,
    feature_cache: dict | None = None,
):
    device = device or next(model.parameters()).device
    sample_feature_cache = _feature_cache_namespace(feature_cache)
    bundle_cache = _bundle_cache_namespace(feature_cache)
    if sample_feature_cache is None:
        sample_feature_cache = {}
    if bundle_cache is None:
        bundle_cache = {}

    decode_cache_mode = _decode_cache_mode(args)
    decode_cache_dir = getattr(args, "decode_cache_dir", None)
    normalized_groups = {}
    sample_keys_by_group = {}
    bundles = {}
    missing_by_key = {}

    for group_name, samples in sample_groups.items():
        normalized_samples = _normalize_samples(samples)
        normalized_groups[group_name] = normalized_samples
        sample_keys = [
            _sample_feature_key(sample, model, args, device, amp_dtype=amp_dtype)
            for sample in normalized_samples
        ]
        sample_keys_by_group[group_name] = sample_keys
        content_key = _bundle_cache_key(sample_keys)
        if content_key in bundle_cache:
            bundles[group_name] = bundle_cache[content_key]
            continue
        for sample_idx, sample_key in enumerate(sample_keys):
            if sample_key not in sample_feature_cache and sample_key not in missing_by_key:
                missing_by_key[sample_key] = normalized_samples[sample_idx]

    if missing_by_key:
        missing_items = list(missing_by_key.items())
        missing_samples = [sample for _, sample in missing_items]
        missing_sample_keys = [sample_key for sample_key, _ in missing_items]
        missing_sample_indices = list(range(len(missing_items)))
        _extract_missing_sample_features(
            model,
            missing_samples,
            transform,
            args,
            amp_dtype=amp_dtype,
            device=device,
            sample_feature_cache=sample_feature_cache,
            sample_keys=missing_sample_keys,
            sample_indices=missing_sample_indices,
            decode_cache_mode=decode_cache_mode,
            decode_cache_dir=decode_cache_dir,
            feature_cache=feature_cache,
        )

    for group_name, normalized_samples in normalized_groups.items():
        if group_name in bundles:
            continue
        sample_keys = sample_keys_by_group[group_name]
        content_key = _bundle_cache_key(sample_keys)
        bundle = _bundle_from_sample_cache(normalized_samples, sample_keys, sample_feature_cache)
        bundle_cache[content_key] = bundle
        bundles[group_name] = bundle

    if feature_cache is not None:
        feature_cache["__sample_feature_cache__"] = sample_feature_cache
        feature_cache["__sample_bundle_cache__"] = bundle_cache
    return bundles


def extract_sample_feature_bundle(
    model,
    samples,
    transform,
    args,
    *,
    amp_dtype=None,
    device=None,
    feature_cache: dict | None = None,
):
    bundle_key = "__single__"
    bundles = extract_sample_feature_bundles(
        model,
        {bundle_key: samples},
        transform,
        args,
        amp_dtype=amp_dtype,
        device=device,
        feature_cache=feature_cache,
    )
    return bundles[bundle_key]


def _bundle_head(bundle, head_index=0):
    feat = bundle["feat"]
    if feat.ndim == 2:
        return feat
    if feat.shape[1] == 1:
        return feat[:, 0, :]
    return feat[:, head_index, :]


def get_pair_head_index(mode1, mode2):
    key = (mode1, mode2)
    if key not in FEATURE_PAIR_TO_HEAD:
        raise ValueError(f"Unsupported feature pair selection: {key}")
    return FEATURE_PAIR_TO_HEAD[key]


def cosine_distance_matrix(q, g, *, device=None, chunk_size=2048, return_tensor=False):
    q = F.normalize(q.float(), p=2, dim=1)
    g = F.normalize(g.float(), p=2, dim=1)
    if q.numel() == 0 or g.numel() == 0:
        empty = torch.empty((q.shape[0], g.shape[0]), dtype=torch.float32)
        return empty if return_tensor else empty.numpy()

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        distmat = -torch.mm(q, g.t())
        return distmat if return_tensor else distmat.cpu().numpy()

    chunk_size = int(chunk_size) if chunk_size and chunk_size > 0 else q.shape[0]
    g_device = g.to(device, non_blocking=True)
    dist_chunks = []
    with torch.inference_mode():
        for start in range(0, q.shape[0], chunk_size):
            stop = min(start + chunk_size, q.shape[0])
            q_chunk = q[start:stop].to(device, non_blocking=True)
            chunk = -torch.mm(q_chunk, g_device.t())
            dist_chunks.append(chunk if return_tensor else chunk.float().cpu())
    distmat = torch.cat(dist_chunks, dim=0)
    return distmat if return_tensor else distmat.numpy()


def _as_torch_float_tensor(array_like, *, device):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array_like, dtype=torch.float32, device=device)


def _as_torch_int_tensor(array_like, *, device):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(device=device, dtype=torch.int32)
    return torch.as_tensor(array_like, dtype=torch.int32, device=device)


def _eval_torch(
    q_pids,
    q_cids,
    q_mids,
    q_camids,
    g_pids,
    g_cids,
    g_mids,
    g_camids,
    mode1,
    mode2,
    distmat,
    dataset=None,
    *,
    device,
    max_rank=20,
):
    with torch.inference_mode():
        distmat = _as_torch_float_tensor(distmat, device=device)
        if distmat.numel() == 0:
            return np.zeros(max_rank, dtype=np.float32), 0.0

        q_pids_t = _as_torch_int_tensor(q_pids, device=device)
        q_cids_t = _as_torch_int_tensor(q_cids, device=device)
        q_mids_t = _as_torch_int_tensor(q_mids, device=device)
        q_camids_t = _as_torch_int_tensor(q_camids, device=device)
        g_pids_t = _as_torch_int_tensor(g_pids, device=device)
        g_cids_t = _as_torch_int_tensor(g_cids, device=device)
        g_mids_t = _as_torch_int_tensor(g_mids, device=device)
        g_camids_t = _as_torch_int_tensor(g_camids, device=device)

        indices = torch.argsort(distmat, dim=1)
        g_pid_sorted = g_pids_t[indices]
        g_cid_sorted = g_cids_t[indices]
        g_mid_sorted = g_mids_t[indices]
        g_camid_sorted = g_camids_t[indices]

        q_pid_col = q_pids_t.view(-1, 1)
        q_cid_col = q_cids_t.view(-1, 1)
        q_mid_col = q_mids_t.view(-1, 1)
        q_camid_col = q_camids_t.view(-1, 1)

        matches = g_pid_sorted.eq(q_pid_col)

        remove_cam = matches & g_camid_sorted.eq(q_camid_col)
        remove_sc = matches & g_cid_sorted.ne(q_cid_col)
        remove_cc = matches & g_cid_sorted.eq(q_cid_col)
        remove_vm = q_mid_col.ne(1) | g_mid_sorted.ne(1)
        remove_im = q_mid_col.ne(2) | g_mid_sorted.ne(2)
        remove_cm = (q_mid_col.eq(1) & g_mid_sorted.eq(1)) | (q_mid_col.eq(2) & g_mid_sorted.eq(2))
        remove_sysu = q_camid_col.eq(3) & g_camid_sorted.eq(2)

        remove = remove_vm if dataset == 'deepchange' else remove_cam
        if dataset == 'sysu':
            remove = remove | remove_sysu

        if mode1 == 'vm':
            remove = remove | remove_vm
        elif mode1 == 'im':
            remove = remove | remove_im
        elif mode1 == 'cm':
            remove = remove | remove_cm

        if mode2 == 'sc':
            remove = remove | remove_sc
        elif mode2 == 'cc':
            remove = remove | remove_cc

        keep = ~remove
        valid_matches = matches & keep
        num_rel = valid_matches.sum(dim=1)
        valid_queries = num_rel.gt(0)
        if not torch.any(valid_queries):
            return np.zeros(max_rank, dtype=np.float32), 0.0

        kept_rank = keep.to(dtype=torch.int32).cumsum(dim=1, dtype=torch.int32)
        kept_count = keep.sum(dim=1, dtype=torch.int32)
        in_top = valid_matches & kept_rank.le(int(max_rank))
        cmc_hits = torch.zeros((distmat.shape[0], int(max_rank)), dtype=torch.float32, device=device)
        rows, cols = torch.where(in_top)
        if rows.numel() > 0:
            rank_cols = kept_rank[rows, cols].to(torch.long) - 1
            cmc_hits[rows, rank_cols] = 1.0
        cmc = cmc_hits.cumsum(dim=1).clamp_max_(1.0)
        rank_positions = torch.arange(1, int(max_rank) + 1, device=device, dtype=torch.int32).view(1, -1)
        cmc = cmc * rank_positions.le(kept_count.view(-1, 1)).to(dtype=torch.float32)
        cmc = cmc[valid_queries].sum(dim=0) / valid_queries.sum().to(dtype=torch.float32)

        precision = valid_matches.cumsum(dim=1, dtype=torch.float32) / kept_rank.clamp_min(1).to(torch.float32)
        ap = (precision * valid_matches.to(torch.float32)).sum(dim=1) / num_rel.clamp_min(1).to(torch.float32)
        mAP = float(ap[valid_queries].mean().item())
        return cmc.detach().cpu().numpy().astype(np.float32, copy=False), mAP


def evaluate_bundle(
    query_bundle,
    gallery_bundle,
    *,
    mode1,
    mode2,
    args,
    default_device,
    dataset=None,
    return_metrics=False,
):
    head_index = 0 if int(getattr(args, "nfeature", 6)) == 1 else get_pair_head_index(mode1, mode2)
    return evaluate_bundle_head(
        query_bundle,
        gallery_bundle,
        head_index=head_index,
        mode1=mode1,
        mode2=mode2,
        args=args,
        default_device=default_device,
        dataset=dataset,
        return_metrics=return_metrics,
    )


def evaluate_bundle_head(
    query_bundle,
    gallery_bundle,
    *,
    head_index,
    mode1,
    mode2,
    args,
    default_device,
    dataset=None,
    return_metrics=False,
):
    q = _bundle_head(query_bundle, head_index=head_index)
    g = _bundle_head(gallery_bundle, head_index=head_index)
    distance_device = resolve_distance_device(args.test_distance_device, default_device=default_device)
    rank_device = resolve_rank_device(
        getattr(args, "test_rank_device", "auto"),
        default_device=distance_device if distance_device.type == "cuda" else default_device,
        num_query=q.shape[0],
        num_gallery=g.shape[0],
        max_cuda_elements=getattr(args, "test_rank_max_elements_cuda", 64000000),
    )

    dist_start = time.perf_counter()
    distmat = cosine_distance_matrix(
        q,
        g,
        device=distance_device,
        chunk_size=getattr(args, "test_distance_chunk_size", 2048),
        return_tensor=rank_device.type == "cuda",
    )
    dist_time = time.perf_counter() - dist_start

    rank_start = time.perf_counter()
    if rank_device.type == "cuda":
        cmc, mAP = _eval_torch(
            query_bundle["pids"],
            query_bundle["cids"],
            query_bundle["mids"],
            query_bundle["camids"],
            gallery_bundle["pids"],
            gallery_bundle["cids"],
            gallery_bundle["mids"],
            gallery_bundle["camids"],
            mode1,
            mode2,
            distmat,
            dataset,
            device=rank_device,
        )
    else:
        cmc, mAP = eval(
            q,
            query_bundle["pids"],
            query_bundle["cids"],
            query_bundle["mids"],
            query_bundle["camids"],
            g,
            gallery_bundle["pids"],
            gallery_bundle["cids"],
            gallery_bundle["mids"],
            gallery_bundle["camids"],
            mode1,
            mode2,
            distmat,
            dataset,
        )
    rank_time = time.perf_counter() - rank_start

    if not return_metrics:
        return cmc, mAP
    return cmc, mAP, {
        "distance_time_sec": float(dist_time),
        "rank_time_sec": float(rank_time),
        "distance_device": distance_device.type,
        "rank_device": rank_device.type,
    }


def eval(q, q_pids, q_cids, q_mids, q_camids,
         g, g_pids, g_cids, g_mids, g_camids,
         mode1, mode2, distmat=None, dataset=None):
    if distmat is None:
        distmat = cosine_distance_matrix(q, g)
    if isinstance(distmat, torch.Tensor):
        distmat = distmat.detach().cpu().numpy()
    num_q, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int64)

    all_cmc = []
    all_ap = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        order = indices[q_idx]
        q_pid = q_pids[q_idx]
        q_cid = q_cids[q_idx]
        q_mid = q_mids[q_idx]
        q_camid = q_camids[q_idx]
        g_pid = g_pids[order]
        g_cid = g_cids[order]
        g_mid = g_mids[order]
        g_camid = g_camids[order]

        remove_cam = (q_pid == g_pid) & (q_camid == g_camid)
        remove_sc = (q_pid == g_pid) & (q_cid != g_cid)
        remove_cc = (q_pid == g_pid) & (q_cid == g_cid)
        remove_vm = (q_mid != 1) | (g_mid != 1)
        remove_im = (q_mid != 2) | (g_mid != 2)
        remove_cm = ((q_mid == 1) & (g_mid == 1)) | ((q_mid == 2) & (g_mid == 2))
        remove_sysu = (q_camid == 3) & (g_camid == 2)

        remove = remove_vm if dataset == 'deepchange' else remove_cam
        if dataset == 'sysu':
            remove = remove | remove_sysu

        if mode1 == 'vm':
            remove = remove | remove_vm
        elif mode1 == 'im':
            remove = remove | remove_im
        elif mode1 == 'cm':
            remove = remove | remove_cm

        if mode2 == 'sc':
            remove = remove | remove_sc
        elif mode2 == 'cc':
            remove = remove | remove_cc

        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if len(orig_cmc) == 0:
            continue
        cmc, ap, num_valid = compute_cmc_ap(orig_cmc)
        if num_valid == 1:
            all_cmc.append(cmc)
            all_ap.append(ap)
            num_valid_q += num_valid

    if num_valid_q == 0:
        return np.zeros(20, dtype=np.float32), 0.0
    all_cmc = np.asarray(all_cmc, dtype=np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_ap))
    return all_cmc, mAP


def compute_cmc_ap(orig_cmc, max_rank=20):
    num_valid = 1 if np.any(orig_cmc) else 0
    cmc = orig_cmc.cumsum()
    cmc[cmc > 1] = 1
    if len(cmc) < max_rank:
        padded = np.zeros(max_rank, dtype=cmc.dtype)
        padded[:len(cmc)] = cmc
        cmc = padded
    else:
        cmc = cmc[:max_rank]
    num_rel = orig_cmc.sum()
    tmp_cmc = orig_cmc.cumsum()
    tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
    ap = tmp_cmc.sum() / max(1, num_rel)
    return cmc, ap, num_valid
