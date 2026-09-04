#!/usr/bin/env python3
"""Run the paper's one-factor-at-a-time GATE-KD sensitivity sweep.

The defaults reproduce the 15K main-results protocol for configuration (c):
Qwen3-Embedding-0.6B -> MiniLMv2-H384, seeds 42/43/44, five epochs, batch 128,
learning rate 7e-5, and the 14,760-row TALAS corpus.  Three knobs are varied one
at a time while every other setting stays at the main configuration:

* topology weight:       0, 0.5, 0.75, 1 (default 1)
* optimizer batch size:  16, 64, 128, 256 (default 128)
* gauge calibration:     2,048, 4,096, 8,192, 16,384 (default 16,384; the
  implementation uses all 14,760 corpus rows when the request exceeds it)

The H0 cloud size always equals the optimizer batch size, including throughout
the batch-size sweep. There is one shared default arm, so the full plan is 10
arms x 3 seeds = 30 jobs, not 12 x 3. A final-test record is the resume boundary.
Retrieval is disabled because it does not change the nine-task Avg. and would
dominate the cost of this sweep.

Examples:
    python scripts/experiments/run_sensitivity.py --dry-run
    python scripts/experiments/run_sensitivity.py
    python scripts/experiments/run_sensitivity.py --sweeps lambda_topo batch_size
    python scripts/experiments/run_sensitivity.py --collect-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src import job_runner

PAIRS = {
    "qwen3_0.6b_to_minilm_h384": {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
        "teacher_pooling": "last_token",
    },
    "bge_m3_to_minilm_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
        "teacher_pooling": "cls",
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
        "teacher_pooling": "last_token",
    },
}

DEFAULT_SEEDS = (42, 43, 44)
DEFAULTS = {
    "lambda_topo": 1.0,
    "batch_size": 128,
    "gauge_samples": 16384,
}
DEFAULT_VALUES = {
    "lambda_topo": (0.0, 0.5, 0.75, 1.0),
    "batch_size": (16, 64, 128, 256),
    "gauge_samples": (2048, 4096, 8192, 16384),
}
SWEEPS = tuple(DEFAULT_VALUES)
SUMMARY_KEYS = ("avg_iod", "avg_ood", "avg_all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pair", choices=sorted(PAIRS), default="qwen3_0.6b_to_minilm_h384"
    )
    parser.add_argument(
        "--train-data",
        default="data/train_set/merged_3_data_5k_each.csv",
        help="default: the 14,760-row 15K corpus",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--sweeps", nargs="+", choices=SWEEPS, default=list(SWEEPS))
    parser.add_argument(
        "--lambda-values", nargs="+", type=float, default=list(DEFAULT_VALUES["lambda_topo"])
    )
    parser.add_argument(
        "--batch-values", nargs="+", type=int, default=list(DEFAULT_VALUES["batch_size"])
    )
    parser.add_argument(
        "--gauge-sample-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_VALUES["gauge_samples"]),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cache-dir", default="runs/teacher_cache")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="jobs per GPU; accuracy is unaffected, timing is not publishable when >1",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--retry-unfinished", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--prewarm-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build/load the shared teacher cache once before parallel fan-out",
    )
    return parser.parse_args(argv)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_root(args: argparse.Namespace) -> Path:
    parent = resolve_path(args.run_root)
    name = args.run_name or f"sensitivity_15k_{args.pair}"
    return parent / name


def _arm_value(value: int | float) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def build_specs(args: argparse.Namespace) -> list[dict]:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be non-empty and unique")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")

    values = {
        "lambda_topo": args.lambda_values,
        "batch_size": args.batch_values,
        "gauge_samples": args.gauge_sample_values,
    }
    specs = [{"arm": "default", "sweep": "default", "value": None, **DEFAULTS}]
    for sweep in args.sweeps:
        if not values[sweep]:
            raise ValueError(f"{sweep} values must not be empty")
        for value in values[sweep]:
            if value == DEFAULTS[sweep]:
                continue
            spec = {"arm": f"{sweep}_{_arm_value(value)}", "sweep": sweep, "value": value, **DEFAULTS}
            spec[sweep] = value
            specs.append(spec)
    names = [spec["arm"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("sensitivity values produce duplicate arm names")
    return specs


def build_command(
    args: argparse.Namespace, spec: dict, seed: int, save_dir: Path
) -> list[str]:
    pair = PAIRS[args.pair]
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--method", "geoode",
        "--train_data", str(resolve_path(args.train_data)),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--student_pooling", "cls",
        "--batch_size", str(spec["batch_size"]),
        "--epochs", str(args.epochs),
        "--save_every", str(args.epochs),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length),
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        "--eval_every", "0",
        "--pair_threshold_source", "test",
        "--cache_dir", str(resolve_path(args.cache_dir)),
        "--save_dir", str(save_dir),
        "--projection_type", "pca",
        "--gauge_align",
        "--gauge_rotation", "procrustes",
        "--gauge_refit_every", "1",
        "--gauge_align_samples", str(spec["gauge_samples"]),
        "--lambda_end", "1.0",
        "--lambda_ctr", "0.0",
        "--lambda_topo", str(spec["lambda_topo"]),
        "--lambda_h1", "0.0",
        # Keep the H0 point cloud identical to the optimizer batch in every arm.
        "--topo_batch_size", str(spec["batch_size"]),
        "--topo_teacher_source", "original",
        "--no_eval_retrieval",
        "--no_wandb",
    ]


def build_jobs(args: argparse.Namespace, specs: list[dict]) -> list[dict]:
    root = run_root(args)
    jobs = []
    for spec in specs:
        for seed in args.seeds:
            save_dir = root / spec["arm"] / f"seed_{seed}"
            jobs.append(
                {
                    "name": f"{spec['arm']}/seed_{seed}",
                    "arm": spec["arm"],
                    "sweep": spec["sweep"],
                    "value": spec["value"],
                    "seed": seed,
                    "run_dir": save_dir,
                    "log_path": save_dir / "train.log",
                    "command": build_command(args, spec, seed, save_dir),
                }
            )
    return jobs


def final_test_record(directory: str | Path) -> dict | None:
    path = Path(directory) / "metrics.jsonl"
    if not path.is_file():
        return None
    found = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("test") and not record.get("train"):
                found = record
    return found


def prewarm_cache(args: argparse.Namespace) -> None:
    pair = PAIRS[args.pair]
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--method", "geoode",
        "--train_data", str(resolve_path(args.train_data)),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--student_pooling", "cls",
        "--batch_size", "128",
        "--epochs", "1",
        "--lr", str(args.lr),
        "--max_length", str(args.max_length),
        "--num_workers", str(args.num_workers),
        "--cache_dir", str(resolve_path(args.cache_dir)),
        "--cache_only",
        "--no_eval_retrieval",
        "--no_wandb",
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpus[0]),
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
    }
    print(f"[cache] {shlex.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def prepare_pending(args: argparse.Namespace, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    pending, status = [], []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for job in jobs:
        directory = Path(job["run_dir"])
        if final_test_record(directory) is not None:
            status.append({"name": job["name"], "status": "skipped_complete"})
            continue
        metrics = directory / "metrics.jsonl"
        if metrics.exists():
            if not args.retry_unfinished:
                raise RuntimeError(
                    f"Unfinished run exists at {directory}; pass --retry-unfinished "
                    "to archive it and retry"
                )
            stale = directory.with_name(f"{directory.name}.stale_{timestamp}")
            directory.rename(stale)
            print(f"[archive] {directory} -> {stale}")
        pending.append(job)
    return pending, status


def execute(args: argparse.Namespace, jobs: list[dict]) -> list[dict]:
    pending, status = prepare_pending(args, jobs)
    if not pending:
        print("All planned jobs already have final-test records.")
        return status

    slots = job_runner.gpu_slots(args.gpus, jobs_per_gpu=args.max_parallel)
    if args.prewarm_cache and len(slots) > 1:
        prewarm_cache(args)
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "TQDM_MININTERVAL": "30",
    }

    def finished(job: dict, row: dict) -> dict:
        complete = row["returncode"] == 0 and final_test_record(job["run_dir"])
        return {**row, "status": "complete" if complete else "failed"}

    rows = job_runner.run_jobs_parallel(
        pending,
        cwd=REPO_ROOT,
        env=env,
        slots=slots,
        poll_seconds=args.poll_seconds,
        stop_on_error=not args.keep_going,
        on_finish=finished,
    )
    status.extend(rows)
    return status


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(args: argparse.Namespace, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seed = []
    for job in jobs:
        record = final_test_record(job["run_dir"])
        row = {
            "arm": job["arm"],
            "sweep": job["sweep"],
            "value": job["value"],
            "seed": job["seed"],
            "status": "done" if record else "missing",
            "run_dir": str(job["run_dir"]),
        }
        if record:
            summary = record["test"].get("summary", {})
            row.update({key: summary.get(key) for key in SUMMARY_KEYS})
        by_seed.append(row)

    grouped = []
    for arm in dict.fromkeys(row["arm"] for row in by_seed):
        rows = [row for row in by_seed if row["arm"] == arm and row["status"] == "done"]
        template = next(row for row in by_seed if row["arm"] == arm)
        out = {
            "arm": arm,
            "sweep": template["sweep"],
            "value": template["value"],
            "n": len(rows),
        }
        for key in SUMMARY_KEYS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            out[f"{key}_mean"] = statistics.mean(values) if values else None
            out[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        grouped.append(out)

    root = run_root(args)
    by_seed_columns = ["arm", "sweep", "value", "seed", "status", *SUMMARY_KEYS, "run_dir"]
    grouped_columns = [
        "arm", "sweep", "value", "n",
        *[f"{key}_{suffix}" for key in SUMMARY_KEYS for suffix in ("mean", "std")],
    ]
    _write_csv(root / "sensitivity_by_seed.csv", by_seed, by_seed_columns)
    _write_csv(root / "sensitivity_mean_std.csv", grouped, grouped_columns)
    done = sum(row["status"] == "done" for row in by_seed)
    print(f"Collected {done}/{len(by_seed)} jobs -> {root}")
    for row in grouped:
        mean, std = row.get("avg_all_mean"), row.get("avg_all_std")
        score = "--" if mean is None else f"{100 * mean:.2f}"
        spread = "--" if std is None else f"{100 * std:.2f}"
        print(f"  {row['arm']:<28} n={row['n']}  Avg={score} +/- {spread}")
    return by_seed, grouped


def main() -> None:
    args = parse_args()
    train_data = resolve_path(args.train_data)
    if not train_data.is_file():
        raise FileNotFoundError(f"Training corpus not found: {train_data}")
    specs = build_specs(args)
    jobs = build_jobs(args, specs)
    print(f"pair:    {args.pair}")
    print(f"corpus:  {train_data}")
    print(f"seeds:   {args.seeds}")
    print(f"output:  {run_root(args)}")
    print(f"plan:    {len(specs)} arms x {len(args.seeds)} seeds = {len(jobs)} jobs")
    for job in jobs:
        print(f"[{job['name']}] {shlex.join(job['command'])}")

    if args.dry_run:
        print("Dry run: no files were written.")
        return
    if args.collect_only:
        collect(args, jobs)
        return

    root = run_root(args)
    root.mkdir(parents=True, exist_ok=True)
    status = execute(args, jobs)
    status_rows = [
        {
            "name": row["name"],
            "status": row["status"],
            "gpu": row.get("gpu"),
            "returncode": row.get("returncode"),
            "seconds": row.get("seconds", 0.0),
        }
        for row in status
    ]
    _write_csv(
        root / "run_status.csv",
        status_rows,
        ["name", "status", "gpu", "returncode", "seconds"],
    )
    collect(args, jobs)
    failed = [row for row in status if row["status"] in {"failed", "not_started"}]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
