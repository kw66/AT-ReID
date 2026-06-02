import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.datamanager import ImageDataset
from others.eval import extract_feature, extract_feature6
from others.transforms import get_transform


ATUSTC_MIX_CASES = (
    {
        "name": "dt-st",
        "label": "DT-ST",
        "query": "q_dt_st",
        "gallery": "g_ad_st",
        "head": 0,
        "moment": "dt",
        "interval": "st",
    },
    {
        "name": "dt-lt",
        "label": "DT-LT",
        "query": "q_dt_lt",
        "gallery": "g_ad_lt",
        "head": 1,
        "moment": "dt",
        "interval": "lt",
    },
    {
        "name": "nt-st",
        "label": "NT-ST",
        "query": "q_nt_st",
        "gallery": "g_ad_st",
        "head": 2,
        "moment": "nt",
        "interval": "st",
    },
    {
        "name": "nt-lt",
        "label": "NT-LT",
        "query": "q_nt_lt",
        "gallery": "g_ad_lt",
        "head": 3,
        "moment": "nt",
        "interval": "lt",
    },
    {
        "name": "ad-st",
        "label": "AD-ST",
        "query": "q_ad_st",
        "gallery": "g_ad_st",
        "head": 4,
        "moment": "ad",
        "interval": "st",
    },
    {
        "name": "ad-lt",
        "label": "AD-LT",
        "query": "q_ad_lt",
        "gallery": "g_ad_lt",
        "head": 5,
        "moment": "ad",
        "interval": "lt",
    },
)

MIX_GROUPS = (
    ("Daytime", ("dt-st", "dt-lt")),
    ("Nighttime", ("nt-st", "nt-lt")),
    ("All-day", ("ad-st", "ad-lt")),
    ("Short-term", ("dt-st", "nt-st", "ad-st")),
    ("Long-term", ("dt-lt", "nt-lt", "ad-lt")),
    ("Anytime", ("dt-st", "dt-lt", "nt-st", "nt-lt", "ad-st", "ad-lt")),
)


def _limit_samples(samples, limit):
    if limit and limit > 0:
        return samples[:limit]
    return samples


def _resolve_feature_mode(args):
    if int(getattr(args, "nfeature", 6)) == 1:
        return "single"
    return str(getattr(args, "test_mix_feature", "adlt")).strip().lower()


def _build_loader(samples, transform, args):
    return DataLoader(
        ImageDataset(samples, transform=transform),
        batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


def _extract_bundle(model, samples, transform, args):
    loader = _build_loader(samples, transform, args)
    if args.nfeature == 1:
        feat, pids, cids, mids, camids = extract_feature(model, loader, args.flip)
    else:
        heads = extract_feature6(model, loader, args.flip)
        feat = torch.stack(heads[:6], dim=1)
        pids, cids, mids, camids = heads[6:]
    return {
        "feat": feat,
        "pids": np.asarray(pids, dtype=np.int64),
        "cids": np.asarray(cids, dtype=np.int64),
        "mids": np.asarray(mids, dtype=np.int64),
        "camids": np.asarray(camids, dtype=np.int64),
    }


def _bundle_fixed_feature(bundle, case, feature_mode):
    feat = bundle["feat"]
    if feat.ndim == 2 or feat.shape[1] == 1:
        return feat if feat.ndim == 2 else feat[:, 0, :]
    if feature_mode == "case":
        return feat[:, case["head"], :]
    if feature_mode == "adlt":
        return feat[:, 5, :]
    if feature_mode == "concat6":
        heads = F.normalize(feat[:, :6, :].float(), p=2, dim=2)
        return heads.reshape(heads.shape[0], -1)
    raise ValueError(f"Unsupported test mix feature mode: {feature_mode}")


def _cosine_distance_matrix(q, g):
    if q.numel() == 0 or g.numel() == 0:
        return np.empty((q.shape[0], g.shape[0]), dtype=np.float32)
    q = F.normalize(q.float(), p=2, dim=1)
    g = F.normalize(g.float(), p=2, dim=1)
    return (-torch.mm(q, g.t())).cpu().numpy()


def _rank(cmc, rank):
    if len(cmc) == 0:
        return 0.0
    return float(cmc[min(rank - 1, len(cmc) - 1)])


def _format_result_line(label, cmc, mAP):
    return (
        f"{label} | R-1: {_rank(cmc, 1):.2%} | R-5: {_rank(cmc, 5):.2%} | "
        f"R-10: {_rank(cmc, 10):.2%} | R-20: {_rank(cmc, 20):.2%} | mAP: {mAP:.2%}"
    )


def _compute_cmc_ap(orig_cmc, max_rank=20):
    cmc = orig_cmc.cumsum()
    cmc[cmc > 1] = 1
    if len(cmc) < max_rank:
        padded = np.zeros(max_rank, dtype=np.float32)
        padded[:len(cmc)] = cmc
        cmc = padded
    else:
        cmc = cmc[:max_rank]
    num_rel = orig_cmc.sum()
    precision = orig_cmc.cumsum() / (np.arange(len(orig_cmc)) + 1.0)
    ap = float((precision * orig_cmc).sum() / max(1, num_rel))
    return cmc.astype(np.float32, copy=False), ap


def _invalid_same_id_mask(case, q_pid, q_cid, q_mid, q_camid, g_pids, g_cids, g_mids, g_camids):
    same_id = g_pids == q_pid
    remove = same_id & (g_camids == q_camid)

    if case["moment"] == "dt":
        remove = remove | (same_id & (g_mids != 1))
    elif case["moment"] == "nt":
        remove = remove | (same_id & (g_mids != 2))
    elif case["moment"] == "ad":
        # AD uses DT+NT galleries; same-ID same-modality images are not valid targets.
        remove = remove | (same_id & (g_mids == q_mid))
    else:
        raise ValueError(f"Unsupported moment: {case['moment']}")

    if case["interval"] == "st":
        remove = remove | (same_id & (g_cids != q_cid))
    elif case["interval"] == "lt":
        remove = remove | (same_id & (g_cids == q_cid))
    else:
        raise ValueError(f"Unsupported interval: {case['interval']}")

    return same_id, remove


def _eval_mixed_case(case, query_bundle, gallery_bundle, feature_mode):
    q = _bundle_fixed_feature(query_bundle, case, feature_mode)
    g = _bundle_fixed_feature(gallery_bundle, case, feature_mode)

    distance_start = time.perf_counter()
    distmat = _cosine_distance_matrix(q, g)
    distance_time = time.perf_counter() - distance_start

    q_pids = np.asarray(query_bundle["pids"])
    q_cids = np.asarray(query_bundle["cids"])
    q_mids = np.asarray(query_bundle["mids"])
    q_camids = np.asarray(query_bundle["camids"])
    g_pids = np.asarray(gallery_bundle["pids"])
    g_cids = np.asarray(gallery_bundle["cids"])
    g_mids = np.asarray(gallery_bundle["mids"])
    g_camids = np.asarray(gallery_bundle["camids"])

    rank_start = time.perf_counter()
    indices = np.argsort(distmat, axis=1, kind="mergesort")
    all_cmc = []
    all_ap = []
    valid_queries = 0
    for q_idx in range(distmat.shape[0]):
        order = indices[q_idx]
        matches, remove = _invalid_same_id_mask(
            case,
            q_pids[q_idx],
            q_cids[q_idx],
            q_mids[q_idx],
            q_camids[q_idx],
            g_pids[order],
            g_cids[order],
            g_mids[order],
            g_camids[order],
        )
        keep = ~remove
        orig_cmc = matches[keep].astype(np.int64, copy=False)
        if len(orig_cmc) == 0 or not np.any(orig_cmc):
            continue
        cmc, ap = _compute_cmc_ap(orig_cmc)
        all_cmc.append(cmc)
        all_ap.append(ap)
        valid_queries += 1
    rank_time = time.perf_counter() - rank_start

    if valid_queries == 0:
        cmc = np.zeros(20, dtype=np.float32)
        mAP = 0.0
    else:
        cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / valid_queries
        mAP = float(np.mean(all_ap))

    return cmc, mAP, {
        "valid_queries": int(valid_queries),
        "distance_time_sec": float(distance_time),
        "rank_time_sec": float(rank_time),
    }


def _weighted_results(results, names):
    total_valid = sum(results[name]["valid_queries"] for name in names)
    if total_valid <= 0:
        return np.zeros(20, dtype=np.float32), 0.0, 0
    cmc = sum(results[name]["cmc"] * results[name]["valid_queries"] for name in names) / total_valid
    mAP = sum(results[name]["map"] * results[name]["valid_queries"] for name in names) / total_valid
    return cmc, float(mAP), int(total_valid)


def _json_ready(results, mixed_results, feature_mode):
    def encode(item):
        return {
            "r1": _rank(item["cmc"], 1),
            "r5": _rank(item["cmc"], 5),
            "r10": _rank(item["cmc"], 10),
            "r20": _rank(item["cmc"], 20),
            "map": float(item["map"]),
            "valid_queries": int(item.get("valid_queries", 0)),
            "cmc": [float(value) for value in item["cmc"]],
        }

    return {
        "feature_mode": feature_mode,
        "protocol": "scenario-agnostic mixed unseen AT-USTC test",
        "atomic": {name: encode(item) for name, item in results.items()},
        "mixed": {
            name: {
                **encode(item),
                "scenarios": list(item["scenarios"]),
            }
            for name, item in mixed_results.items()
        },
    }


def test_mix(args, dataset, model):
    start = time.time()
    _, transform_test = get_transform(args)
    feature_mode = _resolve_feature_mode(args)

    sample_groups = {}
    for case in ATUSTC_MIX_CASES:
        sample_groups[case["query"]] = _limit_samples(getattr(dataset, case["query"]), getattr(args, "limit_query", 0))
        sample_groups[case["gallery"]] = _limit_samples(getattr(dataset, case["gallery"]), getattr(args, "limit_gallery", 0))

    feature_start = time.perf_counter()
    bundles = {}
    for group_name, samples in sample_groups.items():
        bundles[group_name] = _extract_bundle(model, samples, transform_test, args)
    feature_time = time.perf_counter() - feature_start

    print("==> AT-USTC scenario-agnostic mixed unseen tests")
    print("Protocol: mixed galleries keep off-scenario distractors; only invalid same-ID targets are filtered.")
    print("Aggregation: Anytime is valid-query weighted over the six query-gallery relations.")
    print(f"Feature Mode: {feature_mode}")
    results = {}
    distance_time = 0.0
    rank_time = 0.0
    for case in ATUSTC_MIX_CASES:
        cmc, mAP, metrics = _eval_mixed_case(
            case,
            bundles[case["query"]],
            bundles[case["gallery"]],
            feature_mode,
        )
        print(
            f"{_format_result_line(case['label'], cmc, mAP)} | "
            f"query: {case['query']} | gallery: {case['gallery']} | valid_query: {metrics['valid_queries']}"
        )
        distance_time += metrics["distance_time_sec"]
        rank_time += metrics["rank_time_sec"]
        results[case["name"]] = {
            "label": case["label"],
            "query": case["query"],
            "gallery": case["gallery"],
            "cmc": cmc,
            "map": mAP,
            "valid_queries": metrics["valid_queries"],
        }

    print("==> Mixed groups (valid-query weighted)")
    mixed_results = {}
    for label, case_names in MIX_GROUPS:
        cmc, mAP, valid_queries = _weighted_results(results, case_names)
        scenario_text = ",".join(name.upper() for name in case_names)
        print(f"{_format_result_line(label, cmc, mAP)} | scenarios: {scenario_text} | valid_query: {valid_queries}")
        mixed_results[label.lower().replace("-", "_").replace(" ", "_")] = {
            "label": label,
            "scenarios": case_names,
            "cmc": cmc,
            "map": mAP,
            "valid_queries": valid_queries,
        }

    print(f"Feature Time:\t {feature_time:.3f}")
    print(f"Distance Time:\t {distance_time:.3f}")
    print(f"Rank Time:\t {rank_time:.3f}")
    print(f"Evaluation Time:\t {time.time() - start:.3f}")

    if getattr(args, "test_mix_json", ""):
        output_path = Path(args.test_mix_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_ready(results, mixed_results, feature_mode), indent=2), encoding="utf-8")
        print(f"Saved mix metrics: {output_path}")

    anytime = mixed_results["anytime"]
    return anytime["cmc"], anytime["map"]
