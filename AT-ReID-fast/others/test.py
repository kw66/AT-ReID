import time

import numpy as np

from others.eval import evaluate_bundle, extract_sample_feature_bundles
from others.runtime import get_device, select_amp_dtype
from others.transforms import get_transform


ATUSTC_CASES = (
    ("AD-ST VM/SC", "q_ad_st", "g_ad_st", "vm", "sc"),
    ("AD-ST IM/SC", "q_ad_st", "g_ad_st", "im", "sc"),
    ("AD-ST CM/SC", "q_ad_st", "g_ad_st", "cm", "sc"),
    ("AD-LT VM/CC", "q_ad_lt", "g_ad_lt", "vm", "cc"),
    ("AD-LT IM/CC", "q_ad_lt", "g_ad_lt", "im", "cc"),
    ("AD-LT CM/CC", "q_ad_lt", "g_ad_lt", "cm", "cc"),
)


def _limit_samples(samples, limit):
    if limit and limit > 0:
        return samples[:limit]
    return samples


def _format_case_line(label, cmc, mAP):
    return (
        f"{label} | R-1: {cmc[0]:.2%} | R-5: {cmc[4]:.2%} | "
        f"R-10: {cmc[9]:.2%} | R-20: {cmc[19]:.2%} | mAP: {mAP:.2%}"
    )


def _aggregate_results(case_results):
    cmc = np.mean([item["cmc"] for item in case_results], axis=0)
    mAP = float(np.mean([item["map"] for item in case_results]))
    return cmc, mAP


def attest(args, dataset, model):
    start = time.time()
    _, transform_test = get_transform(args, test_pre_resized=str(getattr(args, "decode_cache", "off")).strip().lower() in {"ram", "disk"})
    amp_dtype = select_amp_dtype(args.amp_dtype)
    device = get_device()
    feature_cache = {}
    feature_start = time.perf_counter()

    bundles = extract_sample_feature_bundles(
        model,
        {
            "q_ad_st": _limit_samples(dataset.q_ad_st, args.limit_query),
            "g_ad_st": _limit_samples(dataset.g_ad_st, args.limit_gallery),
            "q_ad_lt": _limit_samples(dataset.q_ad_lt, args.limit_query),
            "g_ad_lt": _limit_samples(dataset.g_ad_lt, args.limit_gallery),
        },
        transform_test,
        args,
        amp_dtype=amp_dtype,
        device=device,
        feature_cache=feature_cache,
    )
    feature_time = time.perf_counter() - feature_start

    case_results = []
    distance_time = 0.0
    rank_time = 0.0
    distance_devices = []
    rank_devices = []
    for label, query_key, gallery_key, mode1, mode2 in ATUSTC_CASES:
        cmc, mAP, metrics = evaluate_bundle(
            bundles[query_key],
            bundles[gallery_key],
            mode1=mode1,
            mode2=mode2,
            args=args,
            default_device=device,
            return_metrics=True,
        )
        print(_format_case_line(label, cmc, mAP))
        distance_time += metrics["distance_time_sec"]
        rank_time += metrics["rank_time_sec"]
        distance_devices.append(metrics["distance_device"])
        rank_devices.append(metrics["rank_device"])
        case_results.append(
            {
                "label": label,
                "cmc": cmc,
                "map": mAP,
            }
        )

    total_time = time.time() - start
    distance_device_summary = "/".join(sorted(set(distance_devices))) if distance_devices else "unknown"
    rank_device_summary = "/".join(sorted(set(rank_devices))) if rank_devices else "unknown"
    print(f"Feature Time:\t {feature_time:.3f}")
    print(f"Distance Time:\t {distance_time:.3f}")
    print(f"Rank Time:\t {rank_time:.3f}")
    print(f"Eval Device:\t distance={distance_device_summary} rank={rank_device_summary}")
    print(f"Evaluation Time:\t {total_time:.3f}")
    avg_cmc, avg_mAP = _aggregate_results(case_results)
    cmc_values = "\t".join(f"{item['cmc'][0] * 100:-2.2f}" for item in case_results)
    map_values = "\t".join(f"{item['map'] * 100:-2.2f}" for item in case_results)
    print(
        f"{avg_cmc[0] * 100:-2.2f}\t{cmc_values}\n"
        f"{avg_mAP * 100:-2.2f}\t{map_values}"
    )
    return avg_cmc, avg_mAP
