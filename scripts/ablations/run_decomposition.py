#!/usr/bin/env python3
"""Run the 15K endpoint/topology decomposition with binary gauge refitting.

The two gauge-refit arms use the same PCA + Procrustes interface.  ``off`` fits
the gauge once at initialization; ``on`` refits it after every epoch.  This is
deliberately different from disabling gauge alignment altogether.

Default plan (configuration c, seeds 42/43/44):

    no_teacher
    h0_only
    endpoint_only
    combined_projected
    combined_original_refit_off
    combined_original_refit_on

That is 6 arms x 3 seeds = 18 jobs.
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

ARMS = (
    {
        "arm": "no_teacher",
        "lambda_end": 0.0,
        "uses_h0": False,
        "h0_source": "original",
        "gauge": False,
        "refit": None,
    },
    {
        "arm": "h0_only",
        "lambda_end": 0.0,
        "uses_h0": True,
        "h0_source": "original",
        "gauge": False,
        "refit": None,
    },
    {
        "arm": "endpoint_only",
        "lambda_end": 1.0,
        "uses_h0": False,
        "h0_source": "original",
        "gauge": True,
        "refit": True,
    },
    {
        "arm": "combined_projected",
        "lambda_end": 1.0,
        "uses_h0": True,
        "h0_source": "projected",
        "gauge": True,
        "refit": True,
    },
    {
        "arm": "combined_original_refit_off",
        "lambda_end": 1.0,
        "uses_h0": True,
        "h0_source": "original",
        "gauge": True,
        "refit": False,
    },
    {
        "arm": "combined_original_refit_on",
        "lambda_end": 1.0,
        "uses_h0": True,
        "h0_source": "original",
        "gauge": True,
        "refit": True,
    },
)

SUMMARY_KEYS = ("avg_iod", "avg_ood", "avg_all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", choices=sorted(PAIRS), default="qwen3_0.6b_to_minilm_h384"
    )
    parser.add_argument(
        "--train-data",
        default="data/train_set/merged_3_data_5k_each.csv",
        help="default: the 14,760-row 15K corpus",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--lambda-topo", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--h0-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gauge-samples", type=int, default=16384)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--probe-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--cache-dir", default="runs/teacher_cache")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--retry-unfinished", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args(argv)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def output_root(args: argparse.Namespace) -> Path:
    name = args.run_name or f"decomposition_15k_{args.pair}"
    return resolve_path(args.run_root) / name


def build_command(
    args: argparse.Namespace, arm: dict, seed: int, save_dir: Path
) -> list[str]:
    pair = PAIRS[args.pair]
    gauge_flags = (
        [
            "--gauge_align",
            "--gauge_rotation",
            "procrustes",
            "--gauge_refit_every",
            "1" if arm["refit"] else "0",
        ]
        if arm["gauge"]
        else ["--no-gauge_align", "--gauge_refit_every", "0"]
    )
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--method",
        "geoode",
        "--train_data",
        str(resolve_path(args.train_data)),
        "--student_model",
        pair["student"],
        "--teacher_model",
        pair["teacher"],
        "--teacher_pooling",
        pair["teacher_pooling"],
        "--student_pooling",
        "cls",
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--save_every",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--max_length",
        str(args.max_length),
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
        "--eval_every",
        "0",
        "--pair_threshold_source",
        "test",
        "--cache_dir",
        str(resolve_path(args.cache_dir)),
        "--save_dir",
        str(save_dir),
        "--projection_type",
        "pca",
        *gauge_flags,
        "--gauge_align_samples",
        str(args.gauge_samples),
        "--lambda_end",
        str(arm["lambda_end"]),
        "--lambda_ctr",
        "0.0",
        "--lambda_topo",
        str(args.lambda_topo if arm["uses_h0"] else 0.0),
        "--lambda_h1",
        "0.0",
        "--topo_batch_size",
        str(args.h0_batch_size),
        "--topo_metric",
        "chord",
        "--topo_teacher_source",
        arm["h0_source"],
        "--probe_every",
        str(args.probe_every),
        "--probe_size",
        str(args.probe_size),
        "--no_eval_retrieval",
        "--no_wandb",
    ]


def build_jobs(args: argparse.Namespace) -> list[dict]:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be non-empty and unique")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    jobs = []
    root = output_root(args)
    for arm in ARMS:
        for seed in args.seeds:
            run_dir = root / arm["arm"] / f"seed_{seed}"
            jobs.append(
                {
                    **arm,
                    "name": f"{arm['arm']}/seed_{seed}",
                    "seed": seed,
                    "run_dir": run_dir,
                    "log_path": run_dir / "train.log",
                    "command": build_command(args, arm, seed, run_dir),
                }
            )
    return jobs


def _last_json_record(path: Path, predicate=None) -> dict | None:
    if not path.is_file():
        return None
    found = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if predicate is None or predicate(record):
                found = record
    return found


def final_test_record(directory: Path) -> dict | None:
    return _last_json_record(
        directory / "metrics.jsonl",
        lambda record: bool(record.get("test")) and not record.get("train"),
    )


def prepare_jobs(args: argparse.Namespace, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    pending, status = [], []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for job in jobs:
        directory = job["run_dir"]
        if final_test_record(directory):
            status.append({"name": job["name"], "status": "skipped_complete"})
            continue
        if (directory / "metrics.jsonl").exists():
            if not args.retry_unfinished:
                raise RuntimeError(
                    f"Unfinished run exists at {directory}; pass --retry-unfinished"
                )
            stale = directory.with_name(f"{directory.name}.stale_{timestamp}")
            directory.rename(stale)
            print(f"[archive] {directory} -> {stale}")
        pending.append(job)
    return pending, status


def prewarm_cache(args: argparse.Namespace) -> None:
    pair = PAIRS[args.pair]
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--method",
        "geoode",
        "--train_data",
        str(resolve_path(args.train_data)),
        "--student_model",
        pair["student"],
        "--teacher_model",
        pair["teacher"],
        "--teacher_pooling",
        pair["teacher_pooling"],
        "--cache_dir",
        str(resolve_path(args.cache_dir)),
        "--cache_only",
        "--no_eval_retrieval",
        "--no_wandb",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpus[0])}
    print(f"[cache] {shlex.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def execute(args: argparse.Namespace, jobs: list[dict]) -> list[dict]:
    pending, status = prepare_jobs(args, jobs)
    if not pending:
        return status
    slots = job_runner.gpu_slots(args.gpus, jobs_per_gpu=args.max_parallel)
    if len(slots) > 1:
        prewarm_cache(args)

    def on_finish(job: dict, row: dict) -> dict:
        complete = row["returncode"] == 0 and final_test_record(job["run_dir"])
        return {**row, "status": "complete" if complete else "failed"}

    rows = job_runner.run_jobs_parallel(
        pending,
        cwd=REPO_ROOT,
        env={**os.environ, "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"},
        slots=slots,
        poll_seconds=args.poll_seconds,
        stop_on_error=not args.keep_going,
        on_finish=on_finish,
    )
    return [*status, *rows]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(args: argparse.Namespace, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seed = []
    for job in jobs:
        final = final_test_record(job["run_dir"])
        probe = _last_json_record(job["run_dir"] / "probe_metrics.jsonl")
        row = {
            "arm": job["arm"],
            "seed": job["seed"],
            "lambda_end": job["lambda_end"],
            "lambda_topo": args.lambda_topo if job["uses_h0"] else 0.0,
            "h0_source": job["h0_source"] if job["uses_h0"] else None,
            "gauge_refit": (
                "on" if job["refit"] else "off" if job["refit"] is False else None
            ),
            "status": "done" if final else "missing",
            "h0_residual": probe.get("probe_h0_w1_teacher") if probe else None,
            "run_dir": str(job["run_dir"]),
        }
        if final:
            summary = final["test"].get("summary", {})
            row.update({key: summary.get(key) for key in SUMMARY_KEYS})
        by_seed.append(row)

    grouped = []
    for arm in (spec["arm"] for spec in ARMS):
        rows = [row for row in by_seed if row["arm"] == arm and row["status"] == "done"]
        template = next(row for row in by_seed if row["arm"] == arm)
        out = {
            key: template[key]
            for key in ("arm", "lambda_end", "lambda_topo", "h0_source", "gauge_refit")
        }
        out["n"] = len(rows)
        for key in (*SUMMARY_KEYS, "h0_residual"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            out[f"{key}_mean"] = statistics.mean(values) if values else None
            out[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        grouped.append(out)

    root = output_root(args)
    by_seed_columns = [
        "arm", "seed", "lambda_end", "lambda_topo", "h0_source", "gauge_refit",
        "status", *SUMMARY_KEYS, "h0_residual", "run_dir",
    ]
    grouped_columns = [
        "arm", "lambda_end", "lambda_topo", "h0_source", "gauge_refit", "n",
        *[
            f"{key}_{suffix}"
            for key in (*SUMMARY_KEYS, "h0_residual")
            for suffix in ("mean", "std")
        ],
    ]
    _write_csv(root / "decomposition_by_seed.csv", by_seed, by_seed_columns)
    _write_csv(root / "decomposition_mean_std.csv", grouped, grouped_columns)
    print(f"Collected {sum(row['status'] == 'done' for row in by_seed)}/{len(jobs)} jobs -> {root}")
    return by_seed, grouped


def main() -> None:
    args = parse_args()
    train_data = resolve_path(args.train_data)
    if not train_data.is_file():
        raise FileNotFoundError(f"Training corpus not found: {train_data}")
    jobs = build_jobs(args)
    print(f"pair:    {args.pair}")
    print(f"corpus:  {train_data}")
    print(f"seeds:   {args.seeds}")
    print(f"output:  {output_root(args)}")
    print(f"plan:    {len(ARMS)} arms x {len(args.seeds)} seeds = {len(jobs)} jobs")
    for job in jobs:
        print(f"[{job['name']}] {shlex.join(job['command'])}")
    if args.dry_run:
        print("Dry run: no files were written.")
        return
    if args.collect_only:
        collect(args, jobs)
        return

    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    status = execute(args, jobs)
    _write_csv(
        root / "run_status.csv",
        status,
        ["name", "status", "gpu", "returncode", "seconds"],
    )
    collect(args, jobs)
    if any(row["status"] in {"failed", "not_started"} for row in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
