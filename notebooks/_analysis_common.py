"""Small shared runtime for the paper notebooks.

The notebooks own experiment-specific settings and plots; this module only keeps
the repeated run/resume/collect code in one tested-looking place.  It deliberately
has no notebook or Colab dependency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

PAIRS = {
    "qwen3_0.6b_to_minilm_h384": {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
        "teacher_pooling": "last_token",
        "student_pooling": "cls",
    },
    "bge_m3_to_minilm_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
        "teacher_pooling": "cls",
        "student_pooling": "cls",
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
        "teacher_pooling": "last_token",
        "student_pooling": "cls",
    },
}

SUMMARY_KEYS = ("avg_iod", "avg_ood", "avg_retrieval", "avg_all")
PAPER_COLORS = {
    "blue": "#2A78D6",
    "orange": "#E07A35",
    "green": "#2A9D6F",
    "red": "#C94C4C",
    "gray": "#7A7A7A",
}


def resolve_project(
    repo_url: str = "https://github.com/duncan-nguyen/embedding-kd.git",
) -> Path:
    """Use the current checkout, or clone it when a notebook runs standalone."""
    cwd = Path.cwd().resolve()
    if (cwd / "main.py").is_file() and (cwd / "distiller.py").is_file():
        return cwd
    parent = Path("/content") if Path("/content").is_dir() else cwd
    project = parent / "embedding-kd"
    if not project.exists():
        subprocess.run(["git", "clone", repo_url, str(project)], check=True)
    if not (project / "main.py").is_file():
        raise FileNotFoundError(f"Invalid project checkout: {project}")
    return project


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(rows)


def final_test_record(run_dir: str | Path) -> dict | None:
    found = None
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("test") and not record.get("train"):
                found = record
    return found


def projection_stats(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "teacher_projection.pt"
    if not path.is_file():
        return {}
    saved = torch.load(path, map_location="cpu", weights_only=False)
    gauge = saved.get("gauge_stats") or {}
    return {
        "projection_type": saved.get("projection_type"),
        "explained_energy": saved.get("explained_energy"),
        "gauge_rotation": saved.get("gauge_rotation")
        if saved.get("gauge_align")
        else "none",
        "gauge_refit_every": saved.get("gauge_refit_every"),
        "gauge_fit_samples": (
            None
            if saved.get("gauge_fit_indices") is None
            else len(saved["gauge_fit_indices"])
        ),
        "gauge_refits": max(0, len(saved.get("gauge_history") or []) - 1),
        "cos_before": gauge.get("cos_before"),
        "cos_after": gauge.get("cos_after"),
        "cos_procrustes": gauge.get("cos_procrustes"),
        "participation_ratio": gauge.get("participation_ratio"),
    }


def distill_command(
    project: Path,
    *,
    method: str,
    pair: dict,
    train_data: Path,
    cache_dir: Path,
    run_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int = 256,
    num_workers: int = 2,
    extra: Iterable[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(project / "main.py"),
        "--method",
        method,
        "--train_data",
        str(train_data),
        "--student_model",
        pair["student"],
        "--teacher_model",
        pair["teacher"],
        "--teacher_pooling",
        pair["teacher_pooling"],
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--save_every",
        str(epochs),
        "--lr",
        str(learning_rate),
        "--max_length",
        str(max_length),
        "--seed",
        str(seed),
        "--num_workers",
        str(num_workers),
        "--eval_every",
        "0",
        "--pair_threshold_source",
        "validation",
        "--no-evaluate_test_each_epoch",
        "--cache_dir",
        str(cache_dir),
        "--save_dir",
        str(run_dir),
        "--no_wandb",
    ]
    if method in {"geoode", "rkd", "simcse"}:
        command.extend(["--student_pooling", pair["student_pooling"]])
    command.extend(map(str, extra))
    return command


def geoode_command(
    project: Path,
    *,
    pair: dict,
    train_data: Path,
    cache_dir: Path,
    run_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int = 256,
    num_workers: int = 2,
    extra: Iterable[str] = (),
) -> list[str]:
    return distill_command(
        project,
        method="geoode",
        pair=pair,
        train_data=train_data,
        cache_dir=cache_dir,
        run_dir=run_dir,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        num_workers=num_workers,
        extra=extra,
    )


def run_jobs(
    project: Path,
    jobs: list[dict],
    *,
    cuda_visible_devices: str = "0",
    stop_on_error: bool = True,
) -> pd.DataFrame:
    """Run jobs sequentially; completed final-test rows are resume boundaries."""
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "TQDM_MININTERVAL": "30",
        }
    )
    status = []
    for index, job in enumerate(jobs, start=1):
        run_dir = Path(job["run_dir"])
        if final_test_record(run_dir) is not None:
            status.append({**job, "status": "skipped_complete", "seconds": 0.0})
            print(f"[SKIP {index}/{len(jobs)}] {job['name']}")
            continue
        if (run_dir / "metrics.jsonl").exists():
            raise RuntimeError(f"Unfinished run exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        print(f"[RUN {index}/{len(jobs)}] {job['name']} -> {run_dir}")
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                job["command"],
                cwd=project,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "%|" not in line:
                    print(line, end="")
            return_code = process.wait()
        seconds = time.perf_counter() - started
        complete = return_code == 0 and final_test_record(run_dir) is not None
        row = {
            **job,
            "status": "complete" if complete else "failed",
            "seconds": seconds,
        }
        status.append(row)
        if not complete and stop_on_error:
            raise RuntimeError(f"Job failed; see {log_path}")
    return pd.DataFrame(status).drop(columns=["command"], errors="ignore")


def collect_jobs(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        row = {
            key: value
            for key, value in job.items()
            if key not in {"command", "run_dir"}
        }
        row["run_dir"] = str(job["run_dir"])
        record = final_test_record(job["run_dir"])
        row["status"] = "done" if record else "missing"
        if record:
            summary = record["test"].get("summary", {})
            row.update({key: summary.get(key) for key in SUMMARY_KEYS})
            row.update(projection_stats(job["run_dir"]))
        rows.append(row)
    return pd.DataFrame(rows)


def teacher_cache_path(
    project: Path,
    cache_dir: Path,
    *,
    pair: dict,
    train_data: Path,
    max_length: int = 256,
) -> Path:
    from src.cache_teacher import cache_filename

    name = cache_filename(
        teacher_model_name=pair["teacher"],
        pooling_method=pair["teacher_pooling"],
        train_data_path=train_data,
        max_length=max_length,
        normalize=True,
    )
    path = Path(cache_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Teacher cache is missing: {path}")
    return path


def load_teacher_cache(path: str | Path) -> tuple[torch.Tensor, dict]:
    from src.cache_teacher import load_cached_embeddings

    return load_cached_embeddings(str(path))


def final_checkpoint(run_dir: str | Path, epochs: int) -> Path:
    path = Path(run_dir) / f"checkpoint_epoch_{epochs}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def set_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
        }
    )
