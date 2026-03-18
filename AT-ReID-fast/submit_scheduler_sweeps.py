from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _default_graphreid_root() -> Path:
    candidates = [
        PROJECT_ROOT.parent / "code",
        Path.home() / "graphreid" / "code",
        Path("/home/lixulin/graphreid/code"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_GRAPHREID_ROOT = _default_graphreid_root()


def _default_data_root_config() -> Path | None:
    candidates = [
        PROJECT_ROOT / "configs" / "dataset_roots.server.json",
        PROJECT_ROOT.parent / "_dataset_roots.server.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_DATA_ROOT_CONFIG = _default_data_root_config()


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    extra_args: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit the final fixed AT-ReID-fast configuration to the GraphReID scheduler.")
    parser.add_argument("--graphreid-root", type=Path, default=DEFAULT_GRAPHREID_ROOT, help="Path to the GraphReID scheduler code root.")
    parser.add_argument("--db", type=Path, default=None, help="Scheduler SQLite path. Defaults to GraphReID's default queue DB.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "scheduler_runs", help="Root directory used by the scheduler for this sweep.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used by queued AT-ReID-fast jobs.")
    parser.add_argument("--dataset", default="atustc")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--data-root-config", type=Path, default=DEFAULT_DATA_ROOT_CONFIG)
    parser.add_argument("--runtime-mode", default="fast-compile", choices=["strict", "fast", "fast-compile"])
    parser.add_argument("--max-epoch", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--test-epoch", type=int, default=999)
    parser.add_argument("--last-test", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--need-mb", type=int, default=7000)
    parser.add_argument("--gpus", default="0,1,2,3", help="Candidate physical GPUs for the queued jobs.")
    parser.add_argument("--tag-prefix", default="atreid_sweep")
    parser.add_argument("--print-only", action="store_true", help="Only print the jobs without submitting them.")
    return parser


def _load_graphreid_modules(graphreid_root: Path):
    graphreid_root = graphreid_root.resolve()
    if not graphreid_root.exists():
        raise FileNotFoundError(f"GraphReID root not found: {graphreid_root}")
    if str(graphreid_root) not in sys.path:
        sys.path.insert(0, str(graphreid_root))
    from graphreid.scheduler.paths import default_db_path
    from graphreid.scheduler.storage import connect_db, create_job, update_job
    from graphreid.scheduler.utils import parse_gpu_spec

    return default_db_path, connect_db, create_job, update_job, parse_gpu_spec


def build_final_full_specs() -> list[ExperimentSpec]:
    return [ExperimentSpec("atreid_fast_full_final", ("-said", "-moae", "-hdw"))]


def build_specs() -> list[ExperimentSpec]:
    return build_final_full_specs()


def build_train_command(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    *,
    run_dir: Path,
    run_version: int,
) -> list[str]:
    model_root = run_dir / "artifacts" / "models"
    log_root = run_dir / "artifacts" / "logs"
    command = [
        str(args.python),
        "-u",
        "train.py",
        "--gpu",
        "env",
        "--dataset",
        args.dataset,
        "--run-version",
        str(run_version),
        "--model-path",
        str(model_root),
        "--log-path",
        str(log_root),
        "--runtime-mode",
        args.runtime_mode,
        "--max-epoch",
        str(args.max_epoch),
        "--workers",
        str(args.workers),
        "--test-epoch",
        str(args.test_epoch),
        "--last-test",
        str(args.last_test),
        "--max-train-steps",
        str(args.max_train_steps),
        "--save-every",
        str(args.save_every),
    ]
    if args.data_root is not None:
        command.extend(["--data-root", str(args.data_root)])
    if args.data_root_config is not None:
        command.extend(["--data-root-config", str(args.data_root_config)])
    if args.test_all:
        command.append("--test-all")
    if args.skip_eval:
        command.append("--skip-eval")
    command.extend(spec.extra_args)
    return command


def main() -> None:
    args = build_parser().parse_args()
    default_db_path, connect_db, create_job, update_job, parse_gpu_spec = _load_graphreid_modules(args.graphreid_root)
    db_path = args.db if args.db is not None else default_db_path()
    candidate_gpus = parse_gpu_spec(args.gpus)
    specs = build_specs()
    if not specs:
        raise SystemExit("No experiments selected.")

    conn = connect_db(db_path)
    args_payload_base = {
        "family": "final-full",
        "dataset": args.dataset,
        "model_name": "atreid_fast",
        "runtime_mode": args.runtime_mode,
        "max_epoch": args.max_epoch,
        "workers": args.workers,
        "test_epoch": args.test_epoch,
        "last_test": args.last_test,
        "max_train_steps": args.max_train_steps,
        "test_all": bool(args.test_all),
        "skip_eval": bool(args.skip_eval),
        "skip_test": True,
        "test_mode": "none",
        "python": str(args.python),
        "scheduler_gpus": candidate_gpus,
        "experiment_count": len(specs),
    }

    for spec in specs:
        tag = f"{args.tag_prefix}_{spec.name}"
        if args.print_only:
            preview_run_dir = args.runs_root / f"preview_{spec.name}"
            preview_command = build_train_command(args, spec, run_dir=preview_run_dir, run_version=0)
            print(f"[DRY-RUN] {tag}")
            print("  " + " ".join(preview_command))
            continue
        placeholder_command = [str(args.python), "-u", "train.py", "--gpu", "env"]
        job = create_job(
            conn,
            job_type="train",
            dataset=args.dataset,
            tag=tag,
            priority=args.priority,
            need_mb=args.need_mb,
            candidate_gpus=candidate_gpus,
            exclusive=False,
            command=placeholder_command,
            env={},
            args_payload={**args_payload_base, "experiment_name": spec.name},
            workdir=PROJECT_ROOT,
            runs_root=args.runs_root,
        )
        run_dir = Path(job["run_dir"])
        command = build_train_command(args, spec, run_dir=run_dir, run_version=int(job["job_id"]))
        job = update_job(
            conn,
            job["job_id"],
            command=command,
            args={
                **args_payload_base,
                "experiment_name": spec.name,
                "command": command,
                "run_version": int(job["job_id"]),
                "model_root": str(run_dir / "artifacts" / "models"),
                "log_root": str(run_dir / "artifacts" / "logs"),
            },
        )
        print(
            f"Submitted job {job['job_id']:04d}  "
            f"tag={tag}  "
            f"run_dir={job['run_dir']}  "
            f"gpus={candidate_gpus if candidate_gpus else 'ALL'}"
        )


if __name__ == "__main__":
    main()
