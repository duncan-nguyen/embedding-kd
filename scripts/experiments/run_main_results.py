#!/usr/bin/env python3
"""Run the main-results sweep: every method x every seed x every teacher/student pair.

This is `notebooks/00_main_results.ipynb` as a headless script, so it can run
unattended over a sweep the notebook only covers one pair at a time. The command
it builds for a job is the notebook's `build_command` flag for flag -- same
epochs, same batch sizes and learning rates, same `--eval_every 0`, same
`--save_every EPOCHS`, same per-pair sub-word markers, same geoode extras -- and
the tables it writes are the notebook's cells 6 and 7. The settings block below
is a copy of the notebook's, which is the one thing to keep in sync: if the
notebook's `METHOD_SETTINGS` or `PAIRS` change, change them here too, or the two
stop being comparable.

    7 methods x 3 seeds x 3 pairs = 63 runs

so it is meant to be started once and left alone. Every job writes its own
directory and a completed job is a resume boundary: re-running the same
--run-name skips what already has a final-test record, which makes a crash, an
OOM or a Ctrl-C cheap to recover from.

Layout under --run-root/<run name>/:

    <pair>/<method>/seed_<seed>/            what main.py writes: metrics.jsonl,
                                            step_metrics.jsonl, checkpoints
    <pair>/<method>/seed_<seed>/train.log   the run's full stdout
    <pair>/<method>/seed_<seed>/runner_timing.json
                                            wall clock of that one run, written
                                            when it finishes, so timings survive
                                            a resume in a later process
    <pair>/final_test_by_seed.csv           one row per (method, seed)
    <pair>/final_test_mean_std{,_paper}.csv mean +- sample std over the seeds
    <pair>/final_test_mean_std.tex
    <pair>/timing_by_seed.csv               training time per method per seed
    <pair>/efficiency_{by_seed,mean_std}.csv
    <pair>/table_3_efficiency.{csv,tex}
    run_status.csv                          appended after every job
    all_pairs_final_test_by_seed.csv        the three pairs stacked
    all_pairs_timing_by_seed.csv
    all_pairs_mean_std.csv

Usage:
    python3 scripts/experiments/run_main_results.py
    python3 scripts/experiments/run_main_results.py --dry-run
    python3 scripts/experiments/run_main_results.py --pairs qwen3_0.6b_to_minilm_h384
    python3 scripts/experiments/run_main_results.py --methods geoode talas --seeds 42
    python3 scripts/experiments/run_main_results.py --run-name <earlier run>   # resume
    python3 scripts/experiments/run_main_results.py --run-name <earlier run> --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Settings. A copy of cell 1 of notebooks/00_main_results.ipynb.
# ---------------------------------------------------------------------------

PAIRS = {
    "qwen3_0.6b_to_minilm_h384": {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
        "teacher_pooling": "last_token",
        "teacher_special_token": "Ġ",
        "emo_teacher_special_token": None,
        "min_vram_gib": 12,
    },
    "bge_m3_to_minilm_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
        "teacher_pooling": "cls",
        "teacher_special_token": "▁",
        "emo_teacher_special_token": "<s>",
        "min_vram_gib": 12,
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
        "teacher_pooling": "last_token",
        "teacher_special_token": "Ġ",
        "emo_teacher_special_token": None,
        "min_vram_gib": 24,
    },
}

DATASETS = {
    "talas_15k": {"path": Path("data/train_set/merged_3_data_5k_each.csv"), "build": None},
    "100k": {"path": Path("data/train_set/train_100k.csv"), "build": None},
    "150k": {
        "path": Path("data/train_set/train_150k.csv"),
        "build": "scripts/data/build_train_corpus.py --total 150000",
    },
    "200k": {
        "path": Path("data/train_set/train_200k.csv"),
        "build": "scripts/data/build_train_corpus.py --total 200000",
    },
}

DEFAULT_DATASET = "100k"
DEFAULT_SEEDS = [42, 43, 44]

MAX_LENGTH = 256
EPOCHS = 5
NUM_WORKERS = 2
CUDA_VISIBLE_DEVICES = "0,1"
# False = one evaluation, straight on the test split (main.py's default).
# True = the held-out protocol: validation every epoch, test once at the end.
HOLD_OUT_VALIDATION = False
EVAL_EVERY = 0
EVAL_RETRIEVAL = False

# Every learned baseline in the main-results table, plus the paper's method. One
# batch size across methods is what makes ms/step and samples/s in the efficiency
# table directly comparable.
METHOD_SETTINGS = {
    "rkd": {"batch_size": 128, "learning_rate": 7e-5},
    "stella": {"batch_size": 128, "learning_rate": 5e-5},
    "cdm": {"batch_size": 128, "learning_rate": 2e-5},
    "dskd": {"batch_size": 128, "learning_rate": 2e-5},
    "emo": {"batch_size": 128, "learning_rate": 1e-5},
    "talas": {"batch_size": 128, "learning_rate": 2e-5},
    "geoode": {"batch_size": 128, "learning_rate": 7e-5},
}

GEOODE_EXTRA = ["--lambda_topo", "1.0", "--lambda_ctr", "0.0", "--gauge_refit_every", "1"]

BENCHMARK_ORDER = [
    "banking77", "tweet", "emotion",
    "mrpc", "scitail", "wic",
    "sick", "sts12", "stsb",
]
SUMMARY_ORDER = ["avg_iod", "avg_ood", "avg_retrieval", "avg_all"]

PROGRESS_EVERY_SEC = 30
PROGRESS_MAX_CHARS = 160


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def build_command(args, pair_name, method, seed, run_dir, cache_dir, train_data):
    """The notebook's build_command, with the pair as a parameter."""
    pair = PAIRS[pair_name]
    settings = METHOD_SETTINGS[method]
    command = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--method", method,
        "--train_data", str(train_data),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--batch_size", str(settings["batch_size"]),
        "--epochs", str(args.epochs),
        "--save_every", str(args.epochs),
        "--lr", str(settings["learning_rate"]),
        "--max_length", str(MAX_LENGTH),
        "--save_dir", str(run_dir),
        "--num_workers", str(args.num_workers),
        "--seed", str(seed),
        "--eval_every", str(EVAL_EVERY),
        "--no_wandb",
    ]
    if args.hold_out_validation:
        command.append("--no-evaluate_test_each_epoch")
    if not args.eval_retrieval:
        command.append("--no_eval_retrieval")
    if method == "cdm":
        command.extend(["--teacher_special_token", pair["teacher_special_token"]])
    if method == "emo" and pair["emo_teacher_special_token"] is not None:
        command.extend(["--teacher_special_token", pair["emo_teacher_special_token"]])
    if method in ("talas", "geoode", "rkd"):
        command.extend(["--cache_dir", str(cache_dir)])
    if method == "geoode":
        command.extend(GEOODE_EXTRA)
    command.extend(settings.get("args", []))
    return command


def build_jobs(args, run_root, cache_dir, train_data):
    """Pair-major, so one pair's teacher cache is built once and each pair
    finishes as a unit rather than all three ending up half done."""
    jobs = []
    for pair_name in args.pairs:
        for method in args.methods:
            for seed in args.seeds:
                run_dir = run_root / pair_name / method / f"seed_{seed}"
                jobs.append({
                    "pair": pair_name,
                    "method": method,
                    "seed": seed,
                    "run_dir": run_dir,
                    "command": build_command(
                        args, pair_name, method, seed, run_dir, cache_dir, train_data
                    ),
                })
    return jobs


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def final_test_record(run_dir):
    """The last record that has test scores and no train block: the end-of-run
    evaluation. Its presence is what makes a job resumable."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return None
    found = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("test") and record.get("train") is None:
                found = record
    return found


def read_timing(run_dir):
    path = Path(run_dir) / "runner_timing.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_timing(run_dir, payload):
    (Path(run_dir) / "runner_timing.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def stream_output(stream, log_handle):
    """Full output to the log, tqdm lines to the console at most every
    PROGRESS_EVERY_SEC so an unattended sweep does not write gigabytes of bar."""
    buffer = ""
    last_progress = 0.0
    progress_shown = False
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buffer += chunk
        parts = buffer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        buffer = parts.pop()
        for line in parts:
            log_handle.write(line + "\n")
            if "%|" in line:
                now = time.perf_counter()
                if now - last_progress >= PROGRESS_EVERY_SEC:
                    print("\r" + line[:PROGRESS_MAX_CHARS].ljust(PROGRESS_MAX_CHARS),
                          end="", flush=True)
                    last_progress = now
                    progress_shown = True
            elif line.strip():
                if progress_shown:
                    print()
                    progress_shown = False
                print(line, flush=True)
        log_handle.flush()
    if buffer:
        log_handle.write(buffer + "\n")
        if "%|" not in buffer:
            print(buffer, flush=True)
    if progress_shown:
        print()


def run_jobs(args, jobs, run_root):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    env["TQDM_MININTERVAL"] = str(PROGRESS_EVERY_SEC)

    status_path = run_root / "run_status.csv"
    status = []
    for position, job in enumerate(jobs, start=1):
        run_dir = job["run_dir"]
        label = f"{job['pair']} / {job['method']} / seed {job['seed']}"

        if final_test_record(run_dir) is not None:
            timing = read_timing(run_dir) or {}
            print(f"[SKIP {position}/{len(jobs)}] {label} — already has a final test")
            status.append({**_job_keys(job), "status": "skipped_complete",
                           "wall_seconds": timing.get("wall_seconds", np.nan)})
            _write_status(status, status_path)
            continue

        if (run_dir / "metrics.jsonl").exists():
            # A run that started and never reached its final evaluation -- it
            # crashed, or the sweep was killed. Appending to it would interleave
            # two runs in one metrics.jsonl, so it has to be moved out of the way
            # before it can be run again.
            if args.retry_unfinished:
                stale = run_dir.with_name(
                    f"{run_dir.name}.stale-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                run_dir.rename(stale)
                print(f"[STALE] moved {run_dir} -> {stale}")
            elif args.stop_on_error:
                raise RuntimeError(
                    f"Unfinished run at {run_dir}. Re-run with --retry-unfinished to "
                    f"move it aside, or delete it yourself."
                )
            else:
                # --keep-going has to mean the same thing on a resume as it does on
                # the first pass, or a sweep that tolerated a failure would abort
                # here on the very run it was told to skip.
                print(f"[SKIP {position}/{len(jobs)}] {label} — unfinished run at "
                      f"{run_dir}; pass --retry-unfinished to redo it")
                status.append({**_job_keys(job), "status": "unfinished_skipped",
                               "wall_seconds": np.nan})
                _write_status(status, status_path)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        print("\n" + "#" * 88)
        print(f"JOB {position}/{len(jobs)}: {label}")
        print(f"Log: {log_path}")
        print("#" * 88, flush=True)

        started_wall = datetime.now(timezone.utc)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                job["command"], cwd=REPO_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert process.stdout is not None
            stream_output(process.stdout, log_handle)
            return_code = process.wait()
        elapsed = time.perf_counter() - started

        complete = return_code == 0 and final_test_record(run_dir) is not None
        write_timing(run_dir, {
            "pair": job["pair"],
            "method": job["method"],
            "seed": job["seed"],
            "status": "complete" if complete else "failed",
            "returncode": return_code,
            "wall_seconds": elapsed,
            "started_at": started_wall.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "cuda_visible_devices": args.cuda_visible_devices,
            "command": shlex.join(job["command"]),
        })
        status.append({**_job_keys(job),
                       "status": "complete" if complete else "failed",
                       "wall_seconds": elapsed})
        _write_status(status, status_path)
        print(f"[{'COMPLETE' if complete else 'FAILED'}] {label} in {elapsed / 60:.1f} min")
        if not complete and args.stop_on_error:
            raise RuntimeError(f"Job failed; see {log_path}")
    return pd.DataFrame(status)


def _job_keys(job):
    return {"pair": job["pair"], "method": job["method"], "seed": job["seed"],
            "run_dir": str(job["run_dir"])}


def _write_status(status, path):
    """Rewritten after every job, so a sweep killed halfway still leaves a
    readable record of what ran and how long it took."""
    frame = pd.DataFrame(status)
    frame["wall_minutes"] = frame["wall_seconds"] / 60
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def benchmark_name(path):
    name = Path(path).stem
    return name[:-5] if name.endswith("_test") else name


def score_from_payload(family, raw_values):
    if family == "classification":
        return float(raw_values["f1"])
    if family == "pair":
        return float(raw_values["average_precision"])
    if family == "sts":
        return float(raw_values)
    if family == "retrieval":
        return float(raw_values["ndcg_at_10"])
    raise KeyError(f"Unknown family: {family}")


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scores_by_seed(args, run_root, pair_name):
    rows, missing = [], []
    for method in args.methods:
        for seed in args.seeds:
            run_dir = run_root / pair_name / method / f"seed_{seed}"
            record = final_test_record(run_dir)
            if record is None:
                missing.append((method, seed, run_dir))
                continue
            payload = record["test"]
            row = {"method": method, "seed": seed}
            for family in ("classification", "pair", "sts", "retrieval"):
                for path, values in payload.get(family, {}).items():
                    row[benchmark_name(path)] = score_from_payload(family, values)
            summary = payload["summary"]
            for key in SUMMARY_ORDER:
                value = summary.get(key)
                row[key] = np.nan if value is None else float(value)
            rows.append(row)
    return pd.DataFrame(rows), missing


def timing_by_seed(args, run_root, pair_name):
    """Two clocks per run, because they answer different questions.

    wall_minutes is what the sweep cost: it includes building the teacher cache
    on the first cached method of a pair, tokenisation and the final evaluation.
    train_gpu_minutes is the sum of step_seconds, so it is the optimisation
    itself and is the one to compare across methods -- the cache is built once
    per pair and would otherwise be charged entirely to whichever method ran
    first.
    """
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            run_dir = run_root / pair_name / method / f"seed_{seed}"
            steps = pd.DataFrame(read_jsonl(run_dir / "step_metrics.jsonl"))
            timed = (
                steps.query("step_seconds > 0 and global_step > 10")
                if not steps.empty else steps
            )
            train_seconds = (
                steps.loc[steps.get("step_seconds", 0) > 0, "step_seconds"].sum()
                if not steps.empty else np.nan
            )
            epoch_records = read_jsonl(run_dir / "metrics.jsonl")
            peaks = [
                record["train"].get("peak_memory_mb", np.nan)
                for record in epoch_records
                if isinstance(record.get("train"), dict)
            ]
            timing = read_timing(run_dir) or {}
            rows.append({
                "pair": pair_name,
                "method": method,
                "seed": seed,
                "wall_minutes": timing.get("wall_seconds", np.nan) / 60
                if timing.get("wall_seconds") is not None else np.nan,
                "train_gpu_minutes": train_seconds / 60,
                "mean_step_ms": 1000 * timed["step_seconds"].mean() if not timed.empty else np.nan,
                "samples_per_second": timed["batch_size"].sum() / timed["step_seconds"].sum()
                if not timed.empty else np.nan,
                "peak_memory_gib": np.nanmax(peaks) / 1024 if peaks else np.nan,
                "started_at": timing.get("started_at"),
                "finished_at": timing.get("finished_at"),
            })
    return pd.DataFrame(rows)


def mean_sd(mean, sd, digits=2):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def aggregate_pair(args, run_root, pair_name):
    """Cells 6 and 7 of the notebook, for one pair. Returns None when the pair is
    not finished: a mean +- std over two of three seeds is not the same table, so
    it is better to report the gap than to publish it."""
    out_dir = run_root / pair_name
    by_seed, missing = scores_by_seed(args, run_root, pair_name)
    timing = timing_by_seed(args, run_root, pair_name)
    timing.to_csv(out_dir / "timing_by_seed.csv", index=False)

    if missing:
        for method, seed, run_dir in missing:
            print(f"  [MISSING] {pair_name} {method} seed={seed}: {run_dir}")
        print(f"  [SKIP AGGREGATE] {pair_name}: {len(missing)} of "
              f"{len(args.methods) * len(args.seeds)} runs have no final test.")
        return None

    metric_order = [n for n in BENCHMARK_ORDER + SUMMARY_ORDER if n in by_seed.columns]
    by_seed = by_seed[["method", "seed", *metric_order]].sort_values(["method", "seed"])
    by_seed.insert(0, "pair", pair_name)

    grouped = by_seed.groupby("method", sort=False)[metric_order]
    means, stds, ns = grouped.mean(), grouped.std(ddof=1), grouped.count()

    wide = {}
    for metric in metric_order:
        wide[f"{metric}_mean"] = means[metric]
        wide[f"{metric}_std"] = stds[metric]
        wide[f"{metric}_n"] = ns[metric]
    mean_std_numeric = pd.DataFrame(wide)

    paper_metrics = [n for n in BENCHMARK_ORDER + ["avg_all"] if n in metric_order]
    paper_display = pd.DataFrame(index=means.index)
    paper_latex = pd.DataFrame(index=means.index)
    for metric in paper_metrics:
        paper_display[metric] = [
            mean_sd(m * 100, s * 100) for m, s in zip(means[metric], stds[metric])
        ]
        paper_latex[metric] = [
            f"{m * 100:.2f} $\\pm$ {s * 100:.2f}" for m, s in zip(means[metric], stds[metric])
        ]

    by_seed.to_csv(out_dir / "final_test_by_seed.csv", index=False)
    mean_std_numeric.to_csv(out_dir / "final_test_mean_std.csv")
    paper_display.to_csv(out_dir / "final_test_mean_std_paper.csv")
    (out_dir / "final_test_mean_std.tex").write_text(
        paper_latex.to_latex(escape=False), encoding="utf-8"
    )

    efficiency = timing.merge(by_seed[["method", "seed", "avg_all"]],
                              on=["method", "seed"], how="left")
    efficiency.to_csv(out_dir / "efficiency_by_seed.csv", index=False)
    summary = efficiency.groupby("method", sort=False).agg(
        avg_mean=("avg_all", "mean"), avg_sd=("avg_all", "std"),
        step_ms_mean=("mean_step_ms", "mean"), step_ms_sd=("mean_step_ms", "std"),
        throughput_mean=("samples_per_second", "mean"), throughput_sd=("samples_per_second", "std"),
        train_min_mean=("train_gpu_minutes", "mean"), train_min_sd=("train_gpu_minutes", "std"),
        wall_min_mean=("wall_minutes", "mean"), wall_min_sd=("wall_minutes", "std"),
        memory_mean=("peak_memory_gib", "mean"), memory_sd=("peak_memory_gib", "std"),
        n=("avg_all", "count"),
    ).reset_index()
    summary.to_csv(out_dir / "efficiency_mean_std.csv", index=False)

    table = pd.DataFrame({
        "Method": summary.method.str.upper(),
        "AVG ↑": [mean_sd(100 * m, 100 * s) for m, s in zip(summary.avg_mean, summary.avg_sd)],
        "ms/step ↓": [mean_sd(m, s, 1) for m, s in zip(summary.step_ms_mean, summary.step_ms_sd)],
        "samples/s ↑": [mean_sd(m, s, 1) for m, s in zip(summary.throughput_mean, summary.throughput_sd)],
        "GPU train min ↓": [mean_sd(m, s, 1) for m, s in zip(summary.train_min_mean, summary.train_min_sd)],
        "wall min": [mean_sd(m, s, 1) for m, s in zip(summary.wall_min_mean, summary.wall_min_sd)],
        "peak GiB ↓": [mean_sd(m, s, 2) for m, s in zip(summary.memory_mean, summary.memory_sd)],
        "n": summary.n,
    })
    table.to_csv(out_dir / "table_3_efficiency.csv", index=False)
    (out_dir / "table_3_efficiency.tex").write_text(
        table.drop(columns="n").to_latex(index=False, escape=False), encoding="utf-8"
    )

    print(f"\n=== {pair_name} — final test, mean ± sample std (x100) ===")
    print(paper_display.to_string())
    print(f"\n=== {pair_name} — efficiency ===")
    print(table.to_string(index=False))

    mean_std_out = mean_std_numeric.reset_index()
    mean_std_out.insert(0, "pair", pair_name)
    return by_seed, mean_std_out, timing


def aggregate(args, run_root):
    all_scores, all_mean_std, all_timing = [], [], []
    for pair_name in args.pairs:
        result = aggregate_pair(args, run_root, pair_name)
        if result is None:
            # Still stack the timings: they exist per run and are useful even for
            # a pair whose scores are not complete yet.
            all_timing.append(timing_by_seed(args, run_root, pair_name))
            continue
        by_seed, mean_std, timing = result
        all_scores.append(by_seed)
        all_mean_std.append(mean_std)
        all_timing.append(timing)

    if all_timing:
        pd.concat(all_timing, ignore_index=True).to_csv(
            run_root / "all_pairs_timing_by_seed.csv", index=False
        )
    if all_scores:
        pd.concat(all_scores, ignore_index=True).to_csv(
            run_root / "all_pairs_final_test_by_seed.csv", index=False
        )
        pd.concat(all_mean_std, ignore_index=True).to_csv(
            run_root / "all_pairs_mean_std.csv", index=False
        )
    print(f"\nWrote aggregates to {run_root}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ensure_data(args, train_data):
    for split in ("train_set", "val_set", "test_set"):
        split_dir = REPO_ROOT / "data" / split
        if not (split_dir.is_dir() and any(split_dir.glob("*.csv"))):
            raise FileNotFoundError(f"Missing data: {split_dir}")

    retrieval_dir = REPO_ROOT / "data" / "test_set" / "retrieval"
    missing_retrieval = [
        name for name in ("arguana", "fiqa", "scidocs")
        if not (retrieval_dir / name / "corpus.csv").is_file()
    ]
    build = DATASETS[args.dataset]["build"]
    if missing_retrieval and (args.eval_retrieval or build):
        if not args.auto_fetch_data:
            raise FileNotFoundError(f"Missing retrieval data: {missing_retrieval}")
        _run_helper(["scripts/data/download_retrieval_benchmarks.py"], "retrieval benchmarks")

    if not train_data.is_file():
        if build is None or not args.auto_fetch_data:
            raise FileNotFoundError(f"Missing training data: {train_data}")
        _run_helper(build.split(), f"corpus {args.dataset}")


def _run_helper(argv, what):
    print(f"[data] {what}: python3 {' '.join(argv)}")
    subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, check=True)


def check_gpu(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible.")
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU must support BF16.")
    largest_gib = 0.0
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 2**30
        largest_gib = max(largest_gib, gib)
        print(f"cuda:{index}: {props.name} ({gib:.1f} GiB)")
    needed = max(PAIRS[name]["min_vram_gib"] for name in args.pairs)
    if largest_gib < needed:
        print(f"[WARN] {needed} GiB on one GPU is recommended for the selected pairs.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs", nargs="+", choices=sorted(PAIRS), default=list(PAIRS),
                        help="teacher/student pairs to sweep (default: all three)")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SETTINGS),
                        default=list(METHOD_SETTINGS),
                        help="methods to run (default: every baseline plus geoode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs",
                        help="parent of the run directory (default: runs/)")
    parser.add_argument("--run-name", default=None,
                        help="name of the run directory; pass an existing one to resume")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="shared teacher cache (default: <run-root>/teacher_cache)")
    parser.add_argument("--cuda-visible-devices", default=CUDA_VISIBLE_DEVICES)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the exact commands, run nothing")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training, rebuild the tables from what is on disk")
    parser.add_argument("--retry-unfinished", action="store_true",
                        help="move an unfinished run aside and start it again")
    parser.add_argument("--keep-going", dest="stop_on_error", action="store_false",
                        help="carry on after a failed job instead of stopping")
    parser.add_argument("--eval-retrieval", action="store_true", default=EVAL_RETRIEVAL)
    parser.add_argument("--hold-out-validation", action="store_true",
                        default=HOLD_OUT_VALIDATION,
                        help="validation every epoch and test once, instead of test only")
    parser.add_argument("--no-auto-fetch-data", dest="auto_fetch_data",
                        action="store_false", help="fail instead of building missing data")
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("--seeds must be at least two distinct values")
    return args


def main(argv=None):
    args = parse_args(argv)

    stamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"main_results_{args.dataset}_{len(args.seeds)}seeds_{stamp}"
    run_root = args.run_root / run_name
    cache_dir = args.cache_dir or (args.run_root / "teacher_cache")
    train_data = REPO_ROOT / DATASETS[args.dataset]["path"]

    jobs = build_jobs(args, run_root, cache_dir, train_data)
    print(f"Run:      {run_root}")
    print(f"Dataset:  {args.dataset} -> {train_data}")
    print(f"Cache:    {cache_dir}")
    print(f"Pairs:    {', '.join(args.pairs)}")
    print(f"Methods:  {', '.join(args.methods)}")
    print(f"Seeds:    {args.seeds}")
    print(f"Plan:     {len(args.pairs)} pairs x {len(args.methods)} methods x "
          f"{len(args.seeds)} seeds = {len(jobs)} jobs")

    if args.dry_run:
        for job in jobs:
            print(f"\n[{job['pair']} / {job['method'].upper()} / seed {job['seed']}]")
            print(shlex.join(job["command"]))
        return 0

    if not args.aggregate_only:
        ensure_data(args, train_data)
        if not args.skip_gpu_check:
            check_gpu(args)
        run_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (run_root / "plan.json").write_text(json.dumps({
            "run_name": run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "train_data": str(train_data),
            "cache_dir": str(cache_dir),
            "epochs": args.epochs,
            "max_length": MAX_LENGTH,
            "pairs": {name: PAIRS[name] for name in args.pairs},
            "method_settings": {m: METHOD_SETTINGS[m] for m in args.methods},
            "seeds": args.seeds,
            "geoode_extra": GEOODE_EXTRA,
            "hold_out_validation": args.hold_out_validation,
            "eval_retrieval": args.eval_retrieval,
            "jobs": [{**_job_keys(job), "command": shlex.join(job["command"])} for job in jobs],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        status = run_jobs(args, jobs, run_root)
        print("\nRun status:")
        for row in status.itertuples():
            print(f"  {row.pair:28s} {row.method:8s} seed={row.seed} "
                  f"{row.status:18s} {row.wall_seconds / 60:8.1f} min")

    if not run_root.is_dir():
        raise FileNotFoundError(f"No such run: {run_root}")
    aggregate(args, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
