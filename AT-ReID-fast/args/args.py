import argparse
from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT_CONFIG = PROJECT_ROOT / "configs" / "dataset_roots.example.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "log"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "save_model"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AT-ReID-fast: a cleaner and faster AT-ReID variant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-d", "--dataset", default="atustc", help="Primary dataset for training or AT-USTC evaluation.")
    parser.add_argument("-v", "--run-version", dest="v", type=int, default=1, help="Version id used in log/model folders.")
    parser.add_argument("-gpu", "--gpu", default="0", help="CUDA_VISIBLE_DEVICES value, for example 0 or 0,1. Use env when launched by an external scheduler.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None, help="Root path for the primary training dataset.")
    parser.add_argument(
        "--data-root-config",
        default=str(DEFAULT_DATA_ROOT_CONFIG),
        help="JSON file that stores dataset roots for AT-USTC and cross-dataset testing.",
    )
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path used for evaluation-only or resume.")
    parser.add_argument("--resume", default=None, help="Resume training from an existing checkpoint.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ViT ImageNet pretrained initialization.",
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="Path to the ViT ImageNet pretrained weight. If omitted, the default torch cache path is used.",
    )

    parser.add_argument("-test", "--test", action="store_true", help="Run AT-USTC evaluation only.")
    parser.add_argument("-test_all", "--test-all", dest="test_all", action="store_true", help="Run cross-dataset evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Alias of --test with clearer semantics for new users.")
    parser.add_argument("--test-batch", type=int, default=256)
    parser.add_argument(
        "--test-batch-auto-tune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Probe several CUDA test batch sizes and reuse the best-throughput choice within the current evaluation call.",
    )
    parser.add_argument(
        "--test-batch-auto-max",
        type=int,
        default=2048,
        help="Upper bound used by --test-batch-auto-tune when searching candidate CUDA test batch sizes.",
    )
    parser.add_argument("--flip", action="store_true", help="Use horizontal flip testing.")
    parser.add_argument("--nfeature", type=int, default=6, choices=[1, 6])
    parser.add_argument("--limit-query", type=int, default=0, help="Use only the first N query samples for fast smoke testing. 0 disables it.")
    parser.add_argument("--limit-gallery", type=int, default=0, help="Use only the first N gallery samples for fast smoke testing. 0 disables it.")
    parser.add_argument(
        "--test-distance-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for query-gallery cosine distance computation.",
    )
    parser.add_argument(
        "--test-distance-chunk-size",
        type=int,
        default=2048,
        help="Chunk size used when cosine distance is computed on GPU.",
    )
    parser.add_argument(
        "--test-rank-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by exact sorting / CMC / mAP evaluation after the distance matrix is built.",
    )
    parser.add_argument(
        "--test-rank-sort-batch-size",
        type=int,
        default=256,
        help="Query block size used by the exact CUDA ranking backend.",
    )
    parser.add_argument(
        "--test-rank-auto-min-pairs",
        type=int,
        default=5000000,
        help="Auto mode prefers CUDA ranking once query_count * gallery_count reaches this value.",
    )
    parser.add_argument(
        "--test-rank-max-elements-cuda",
        type=int,
        default=1200000000,
        help="Hard upper bound for auto CUDA ranking. Set <= 0 to remove the cap.",
    )
    parser.add_argument(
        "--decode-cache",
        default=None,
        choices=["off", "ram", "disk"],
        help="Optional decoded-image cache used by evaluation loaders. Current runtime presets keep it off by default.",
    )
    parser.add_argument(
        "--decode-cache-dir",
        default=None,
        help="Optional cache root used when --decode-cache is enabled. Defaults to an inferred per-dataset cache folder.",
    )

    parser.add_argument(
        "--runtime-mode",
        default="fast-compile",
        choices=["strict", "fast", "fast-compile"],
        help="fast-compile is the default long-run preset; fast skips compile; strict keeps the old conservative path.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Enable AMP for training.")
    parser.add_argument(
        "--amp-dtype",
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="AMP dtype for training. auto prefers bf16 when supported.",
    )
    parser.add_argument(
        "--test-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable AMP during feature extraction and evaluation.",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="Enable torch.compile.")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode used when --compile is enabled.",
    )
    parser.add_argument(
        "--optimizer-fused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fused optimizer if the selected optimizer supports it.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use deterministic cudnn behavior. strict defaults to true, fast modes default to false.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul and cudnn kernels on Ampere+ GPUs.",
    )
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=["highest", "high", "medium"],
        help="torch.set_float32_matmul_precision setting.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pin_memory in dataloaders.",
    )
    parser.add_argument(
        "--non-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use non_blocking=True for cuda copies when possible.",
    )
    parser.add_argument(
        "--train-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use persistent workers for the training dataloader when workers>0.",
    )
    parser.add_argument(
        "--test-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use persistent workers for evaluation dataloaders when workers>0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="prefetch_factor used when num_workers>0.",
    )
    parser.add_argument(
        "--zero-grad-set-to-none",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call optimizer.zero_grad(set_to_none=True) for a slightly lighter training step.",
    )
    parser.add_argument(
        "--inference-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.inference_mode during feature extraction.",
    )

    parser.add_argument("--drop", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("-ncls", "--ncls", type=int, default=6, choices=[1, 6])
    parser.add_argument("-said", "--said", action="store_true")
    parser.add_argument(
        "-hdw",
        "--hdw",
        action="store_true",
        help="Enable the fixed fast HDW weighting used by AT-ReID-fast. Ablation interfaces have been removed.",
    )
    parser.add_argument(
        "-moae",
        "--moae",
        action="store_true",
        help="Enable the fixed final MOAE variant used by AT-ReID-fast. Experimental MOAE interfaces have been removed.",
    )
    parser.add_argument(
        "--vit-attention-backend",
        default="auto",
        choices=["auto", "fused", "math", "eager"],
        help="Attention backend for the custom ViT attention. auto/fused use SDPA when available.",
    )
    parser.add_argument("--optim", default="sgd", choices=["sgd", "adam", "adamW"])
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--dwd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dlr", type=float, default=10)
    parser.add_argument("--dlrl", type=int, nargs="*", default=[10, 11, 12])
    parser.add_argument("--clip", type=float, default=20)
    parser.add_argument("--warmup-loss", dest="warmup_loss", type=int, default=999)
    parser.add_argument("--warmup-rate", type=float, default=0.1)
    parser.add_argument("--warmup-epoch", type=int, default=10)
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--scheduler-epoch", type=int, nargs="*", default=[])
    parser.add_argument("--cosmin-rate", type=float, default=0.002)
    parser.add_argument("--max-epoch", type=int, default=120)
    parser.add_argument("--era", type=float, default=0.4)
    parser.add_argument("--gray", type=float, default=0.0)
    parser.add_argument("--ih", type=int, default=256)
    parser.add_argument("--iw", type=int, default=128)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sample", default="m")
    parser.add_argument("--md1", type=int, nargs="*", default=[1, 3, 2, 0])
    parser.add_argument("--md2", type=int, nargs="*", default=[1, 3, 0, 0])
    parser.add_argument("--d1", type=float, default=0.6)
    parser.add_argument("--d2", type=float, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--test-epoch", type=int, default=999)
    parser.add_argument("--last-test", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0, help="Also save epoch_{epoch:03d}.t every N epochs. 0 disables it.")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop each epoch after this many training iterations. 0 means full epoch.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip validation/testing during and after training. Useful for fast smoke tests.")
    parser.add_argument(
        "--train-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect and persist per-epoch HDW/MOAE diagnostic statistics. Disabled by default to avoid extra training overhead.",
    )

    return parser


def apply_runtime_preset(args) -> None:
    if args.runtime_mode == "strict":
        if args.amp is None:
            args.amp = False
        if args.test_amp is None:
            args.test_amp = False
        if args.compile is None:
            args.compile = False
        if args.deterministic is None:
            args.deterministic = True
        if args.train_persistent_workers is None:
            args.train_persistent_workers = False
        if args.test_persistent_workers is None:
            args.test_persistent_workers = False
        if args.test_batch_auto_tune is None:
            args.test_batch_auto_tune = False
        if args.decode_cache is None:
            args.decode_cache = "off"
    elif args.runtime_mode == "fast":
        if args.amp is None:
            args.amp = True
        if args.test_amp is None:
            args.test_amp = True
        if args.compile is None:
            args.compile = False
        if args.deterministic is None:
            args.deterministic = False
        if args.train_persistent_workers is None:
            args.train_persistent_workers = True
        if args.test_persistent_workers is None:
            args.test_persistent_workers = True
        if args.test_batch_auto_tune is None:
            args.test_batch_auto_tune = False
        if args.decode_cache is None:
            args.decode_cache = "off"
    elif args.runtime_mode == "fast-compile":
        if args.amp is None:
            args.amp = True
        if args.test_amp is None:
            args.test_amp = True
        if args.compile is None:
            args.compile = True
        if args.deterministic is None:
            args.deterministic = False
        if args.train_persistent_workers is None:
            args.train_persistent_workers = True
        if args.test_persistent_workers is None:
            args.test_persistent_workers = True
        if args.test_batch_auto_tune is None:
            args.test_batch_auto_tune = False
        if args.decode_cache is None:
            args.decode_cache = "off"

    if args.eval_only:
        args.test = True


def normalize_paths(args) -> None:
    args.gpu = str(args.gpu)
    if args.data_root is not None:
        args.data_root = str(Path(args.data_root))
    if args.data_root_config is not None:
        args.data_root_config = str(Path(args.data_root_config))
    if args.pretrained_path is not None:
        args.pretrained_path = str(Path(args.pretrained_path))
    if args.decode_cache_dir is not None:
        args.decode_cache_dir = str(Path(args.decode_cache_dir))
    if args.checkpoint is not None:
        args.checkpoint = str(Path(args.checkpoint))
    if args.resume is not None:
        args.resume = str(Path(args.resume))


def create_argparser() -> argparse.ArgumentParser:
    return build_parser()


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_runtime_preset(args)
    normalize_paths(args)
    if (
        list(args.md1) != [1, 3, 2, 0]
        or list(args.md2) != [1, 3, 0, 0]
        or float(args.d1) != 0.6
        or float(args.d2) != 6.0
    ):
        warnings.warn(
            "AT-ReID-fast keeps --md1/--md2/--d1/--d2 only for CLI compatibility. "
            "This simplified fast folder does not consume those legacy relation-loss arguments.",
            stacklevel=2,
        )
    return args


if __name__ == "__main__":
    print(parse_args())
