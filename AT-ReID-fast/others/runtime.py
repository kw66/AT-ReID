from contextlib import nullcontext
import os
from pathlib import Path
import warnings

import torch


def ensure_compile_cache_dirs(base_dir: str | Path | None = None) -> dict[str, str]:
    root = Path(base_dir or os.environ.get("ATREID_FAST_RUNTIME_CACHE_DIR", Path.home() / ".cache" / "atreid-fast")).expanduser().resolve()
    inductor = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", root / "torchinductor")).expanduser().resolve()
    triton = Path(os.environ.get("TRITON_CACHE_DIR", root / "triton")).expanduser().resolve()
    inductor.mkdir(parents=True, exist_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    return {
        "root": str(root),
        "torchinductor": str(inductor),
        "triton": str(triton),
    }


def setup_runtime(args):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if bool(getattr(args, "compile", False)):
        return ensure_compile_cache_dirs()
    return None


def select_amp_dtype(amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(enabled: bool, dtype=None, device_type: str = "cuda"):
    if not enabled or device_type != "cuda" or not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def create_grad_scaler(enabled: bool, amp_dtype):
    if not enabled or amp_dtype != torch.float16 or not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def maybe_compile_model(model, args):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile is not available in this torch version. Falling back to eager mode.")
        return model
    try:
        return torch.compile(model, mode=args.compile_mode)
    except Exception as exc:  # pragma: no cover - environment-specific
        warnings.warn(f"torch.compile failed with {exc!r}. Falling back to eager mode.")
        return model


def build_dataloader_kwargs(args, train: bool):
    kwargs = {
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "drop_last": train,
    }
    if args.workers > 0:
        persistent_workers = args.train_persistent_workers if train else args.test_persistent_workers
        kwargs["persistent_workers"] = bool(persistent_workers)
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_distance_device(requested: str, *, default_device: torch.device) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return default_device if default_device.type == "cuda" else torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported test distance device: {requested}")


def resolve_rank_device(
    requested: str,
    *,
    default_device: torch.device,
    num_query: int,
    num_gallery: int,
    max_cuda_elements: int,
) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        total = int(num_query) * int(num_gallery)
        if default_device.type == "cuda" and torch.cuda.is_available() and total <= int(max_cuda_elements):
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported test rank device: {requested}")


def resolve_vit_attention_backend(requested: str, *, use_sdpa_available: bool) -> dict[str, object]:
    backend = str(requested).strip().lower()
    if backend not in {"auto", "fused", "math", "eager"}:
        raise ValueError(f"Unsupported ViT attention backend: {requested}")
    cuda_backends = getattr(torch.backends, "cuda", None)

    def _optional_backend_flag(name: str):
        fn = getattr(cuda_backends, name, None) if cuda_backends is not None else None
        if not callable(fn):
            return None
        try:
            return bool(fn())
        except Exception:  # pragma: no cover - backend API varies by torch build
            return None

    device_capability = None
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            device_capability = f"{major}.{minor}"
        except Exception:  # pragma: no cover - environment-specific
            device_capability = None

    if backend == "eager":
        return {
            "requested": backend,
            "active": "eager",
            "uses_sdpa": False,
            "has_sdpa": bool(use_sdpa_available),
            "flash_sdp_enabled": _optional_backend_flag("flash_sdp_enabled"),
            "mem_efficient_sdp_enabled": _optional_backend_flag("mem_efficient_sdp_enabled"),
            "math_sdp_enabled": _optional_backend_flag("math_sdp_enabled"),
            "cudnn_sdp_enabled": _optional_backend_flag("cudnn_sdp_enabled"),
            "device_capability": device_capability,
        }
    if not use_sdpa_available:
        return {
            "requested": backend,
            "active": "eager",
            "uses_sdpa": False,
            "has_sdpa": False,
            "flash_sdp_enabled": _optional_backend_flag("flash_sdp_enabled"),
            "mem_efficient_sdp_enabled": _optional_backend_flag("mem_efficient_sdp_enabled"),
            "math_sdp_enabled": _optional_backend_flag("math_sdp_enabled"),
            "cudnn_sdp_enabled": _optional_backend_flag("cudnn_sdp_enabled"),
            "device_capability": device_capability,
        }
    if backend == "math":
        active = "math"
    elif backend == "fused":
        active = "fused"
    else:
        active = "sdpa"
    return {
        "requested": backend,
        "active": active,
        "uses_sdpa": True,
        "has_sdpa": True,
        "flash_sdp_enabled": _optional_backend_flag("flash_sdp_enabled"),
        "mem_efficient_sdp_enabled": _optional_backend_flag("mem_efficient_sdp_enabled"),
        "math_sdp_enabled": _optional_backend_flag("math_sdp_enabled"),
        "cudnn_sdp_enabled": _optional_backend_flag("cudnn_sdp_enabled"),
        "device_capability": device_capability,
    }


def format_vit_attention_info(info: dict | None) -> str:
    if not info:
        return "unavailable"
    parts = [
        f"requested={info.get('requested', '-')}",
        f"active={info.get('active', '-')}",
        f"uses_sdpa={bool(info.get('uses_sdpa', False))}",
        f"has_sdpa={bool(info.get('has_sdpa', False))}",
    ]
    if info.get("flash_sdp_enabled") is not None:
        parts.append(f"flash_sdp={bool(info['flash_sdp_enabled'])}")
    if info.get("mem_efficient_sdp_enabled") is not None:
        parts.append(f"mem_eff_sdp={bool(info['mem_efficient_sdp_enabled'])}")
    if info.get("math_sdp_enabled") is not None:
        parts.append(f"math_sdp={bool(info['math_sdp_enabled'])}")
    if info.get("device_capability") is not None:
        parts.append(f"sm={info['device_capability']}")
    return ", ".join(parts)


def sdpa_context(attention_backend: str):
    backend = str(attention_backend).strip().lower()
    cuda_backends = getattr(torch.backends, "cuda", None)
    sdp_kernel = getattr(cuda_backends, "sdp_kernel", None) if cuda_backends is not None else None
    if not callable(sdp_kernel):
        return nullcontext()
    if backend == "math":
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    if backend == "fused":
        return sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    return nullcontext()
