#!/usr/bin/env bash
# Score every un-distilled model -- the three teachers and the three student
# initialisations of the main-results grid -- on the same test benchmarks, three
# seeds each. This is the "row zero" of the main table: what the student knows
# before any distillation, and what the teacher it is chasing scores.
#
#     bash scripts/experiments/run_baseline_eval.sh
#     MODELS="Qwen/Qwen3-Embedding-0.6B" SEEDS=42 bash scripts/experiments/run_baseline_eval.sh
#
# There is no training here: one forward pass over each benchmark plus the
# logistic probe of the classification group, scored through src/evaluation, so
# the numbers drop straight into the main-results table. The seeds pin every RNG
# of the process but nothing in the pipeline is stochastic, so the three seeds
# agree to the digit and the std column reads 0.00 -- that zero is the evidence
# there is no variance to report, not a measurement of it. SEEDS=42 buys the same
# conclusion at a third of the compute.
#
# One (model, seed) writes one results.json and a re-run skips whatever is
# already on disk, so an interrupted sweep resumes by being started again with
# the same RUN_NAME.
#
# Environment:
#     MODELS                space-separated subset of the six baselines below
#     SEEDS                 space-separated seeds        (default: 42 43 44)
#     EVAL_RETRIEVAL        1 scores the five retrieval benchmarks (~101k docs
#                           embedded per model), 0 skips them   (default: 1)
#     RUN_NAME              resume an earlier sweep by name
#     RUN_ROOT              parent of the run directory  (default: runs/)
#     STOP_ON_ERROR         1 aborts on the first failure, 0 carries on (default: 1)
#     CUDA_VISIBLE_DEVICES  devices for the eval process (default: 0)
#     PYTHON                interpreter (default: .venv/bin/python, else python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Teachers first, then the student initialisations. The pooling is the model's
# own sentence vector -- Qwen3-Embedding is a decoder and reads its last token,
# an encoder reads "cls" -- and is the same value the distillation runs use, so
# a baseline is scored through the same interface its distilled run is.
MODELS="${MODELS:-\
Qwen/Qwen3-Embedding-0.6B \
Qwen/Qwen3-Embedding-4B \
BAAI/bge-m3 \
nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base \
nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base \
google-bert/bert-base-uncased}"

SEEDS="${SEEDS:-42 43 44}"
EVAL_RETRIEVAL="${EVAL_RETRIEVAL:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    python_bin="$REPO_ROOT/.venv/bin/python"
else
    python_bin="python3"
fi

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
seed_count="$(wc -w <<< "$SEEDS")"
RUN_NAME="${RUN_NAME:-baselines_${seed_count// /}seeds_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"
log_path="$RUN_DIR/eval.log"

resume_hint() {
    echo
    echo "Baseline eval stopped. Everything finished so far is kept; resume with:"
    echo "    RUN_NAME=$RUN_NAME bash scripts/experiments/run_baseline_eval.sh"
    echo "Full log: $log_path"
}
trap resume_hint ERR INT TERM

echo "======================================"
echo "Baseline eval — teacher/student base models"
echo "  run:       $RUN_DIR"
echo "  log:       $log_path"
echo "  python:    $python_bin"
echo "  seeds:     $SEEDS"
echo "  retrieval: $EVAL_RETRIEVAL"
echo "  started    $(date -Is)"
echo "======================================"

cd "$REPO_ROOT"

# The retrieval corpora are downloaded, not in git; the sentence-level splits are
# expected to be there already, the same as every training script assumes.
if [[ "$EVAL_RETRIEVAL" == "1" ]]; then
    if [[ ! -f data/test_set/retrieval/nfcorpus/corpus.csv ]]; then
        echo "Retrieval benchmarks missing — downloading."
        "$python_bin" scripts/data/download_retrieval_benchmarks.py
    fi
fi

export MODELS SEEDS EVAL_RETRIEVAL STOP_ON_ERROR RUN_DIR

# Appended, not truncated: a resume of the same run keeps the earlier passes.
"$python_bin" - <<'PY' 2>&1 | tee -a "$log_path"
"""Eval each (model, seed) once, then aggregate every results.json in the run."""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import __version__ as transformers_version

sys.path.insert(0, str(Path.cwd()))

from distiller import KnowledgeDistiller
from src.evaluation.evaluation_automodel import (
    eval_classification_task,
    eval_pair_task,
    eval_sts_task,
    test_cls_tasks,
    test_pair_tasks,
    test_sts_tasks,
)
from src.evaluation.retrieval import eval_retrieval_task, test_retrieval_tasks
from src.pooling import pool_sentence_embedding

# Pooling and load dtype per baseline: the teachers are served in bfloat16 the
# way the cache builds them, the students in float32 the way training holds them.
BASELINES = {
    "Qwen/Qwen3-Embedding-0.6B": {"pooling": "last_token", "dtype": "bfloat16"},
    "Qwen/Qwen3-Embedding-4B": {"pooling": "last_token", "dtype": "bfloat16"},
    "BAAI/bge-m3": {"pooling": "cls", "dtype": "bfloat16"},
    "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base": {"pooling": "cls", "dtype": "float32"},
    "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base": {"pooling": "cls", "dtype": "float32"},
    "google-bert/bert-base-uncased": {"pooling": "cls", "dtype": "float32"},
}

BENCHMARK_ORDER = [
    "banking77", "tweet", "emotion",
    "mrpc", "scitail", "wic",
    "sick", "sts12", "stsb",
]
SUMMARY_ORDER = ["avg_iod", "avg_ood", "avg_retrieval", "avg_all"]

# transformers >= 5 renamed the load-dtype keyword.
DTYPE_ARG = "dtype" if int(transformers_version.split(".")[0]) >= 5 else "torch_dtype"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

RUN_DIR = Path(os.environ["RUN_DIR"])
MODELS = os.environ["MODELS"].split()
SEEDS = [int(seed) for seed in os.environ["SEEDS"].split()]
EVAL_RETRIEVAL = os.environ["EVAL_RETRIEVAL"] == "1"
STOP_ON_ERROR = os.environ["STOP_ON_ERROR"] == "1"

unknown = [name for name in MODELS if name not in BASELINES]
if unknown:
    raise SystemExit(
        f"Unknown baseline(s): {unknown}\nKnown: {list(BASELINES)}"
    )
if len(set(SEEDS)) != len(SEEDS):
    raise SystemExit(f"SEEDS must not repeat: {SEEDS}")
DEVICE = None


class PooledEncoder(torch.nn.Module):
    """AutoModel plus that model's own pooling.

    ``_embed_texts`` reads the "pooled" key when a forward returns a dict, so a
    decoder teacher is scored at its last token rather than forced onto the CLS
    position an encoder student uses.
    """

    def __init__(self, name, pooling, dtype):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            name, trust_remote_code=True, **{DTYPE_ARG: DTYPES[dtype]}
        )
        self.pooling = pooling

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_sentence_embedding(
            output.last_hidden_state, attention_mask, self.pooling
        )
        return {"pooled": pooled}


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


def evaluate_baseline(name, settings, seed):
    # Nothing here is trained, so there is no seed to consume; pinning every RNG
    # of the process anyway is what makes two seeds disagreeing worth a look.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoder = PooledEncoder(name, settings["pooling"], settings["dtype"]).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        # The pair group sweeps its threshold on test -- there is no validation run
        # to inherit one from -- which is main.py's own pair_threshold_source="test".
        pair, _ = eval_pair_task(encoder, test_pair_tasks, tokenizer)
        results = {
            "classification": eval_classification_task(encoder, test_cls_tasks, tokenizer),
            "pair": pair,
            "sts": eval_sts_task(encoder, test_sts_tasks, tokenizer),
            "retrieval": (
                eval_retrieval_task(encoder, test_retrieval_tasks, tokenizer)
                if EVAL_RETRIEVAL else {}
            ),
            "pair_threshold_source": "test",
        }
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    scores = {
        KnowledgeDistiller._benchmark_name(path, "test"): score_from_payload(family, values)
        for family in ("classification", "pair", "sts", "retrieval")
        for path, values in results[family].items()
    }
    averages = KnowledgeDistiller._benchmark_group_averages(scores)
    results["summary"] = {key: group["score"] for key, group in averages.items()}
    return results


jobs = [(name, seed) for name in MODELS for seed in SEEDS]


def result_path_of(name, seed):
    return RUN_DIR / name.replace("/", "__") / f"seed_{seed}" / "results.json"


# The card is only needed for work still to do, so a re-run of a finished sweep
# reprints its table on any machine.
pending = [(name, seed) for name, seed in jobs if not result_path_of(name, seed).is_file()]
if pending:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible; baseline eval needs a GPU.")
    DEVICE = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    print(f"cuda:0: {properties.name} ({properties.total_memory / 2**30:.1f} GiB)")
else:
    print("Every (model, seed) already has results.json; aggregating only.")

status = []
for position, (name, seed) in enumerate(jobs, start=1):
    result_path = result_path_of(name, seed)
    if result_path.is_file():
        print(f"[SKIP] {name} seed={seed} already has results.json")
        status.append({"model": name, "seed": seed, "status": "skipped_complete", "seconds": 0.0})
        continue
    print("\n" + "#" * 88)
    print(f"JOB {position}/{len(jobs)}: {name} — seed {seed}")
    print("#" * 88, flush=True)
    started = time.perf_counter()
    try:
        results = evaluate_baseline(name, BASELINES[name], seed)
    except Exception as error:
        elapsed = time.perf_counter() - started
        print(f"[FAILED] {name} seed={seed}: {error}")
        status.append({"model": name, "seed": seed, "status": "failed", "seconds": elapsed})
        if STOP_ON_ERROR:
            raise
        continue
    elapsed = time.perf_counter() - started
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"model": name, "seed": seed, **BASELINES[name], "test": results}, indent=2),
        encoding="utf-8",
    )
    status.append({"model": name, "seed": seed, "status": "complete", "seconds": elapsed})
    print(f"[COMPLETE] {name} seed={seed} in {elapsed / 60:.1f} min -> {result_path}")

print("\nStatus:")
for item in status:
    print(f"  {item['model']:55s} seed={item['seed']} {item['status']:18s} {item['seconds'] / 60:8.1f} min")

# Aggregate whatever is on disk, so a resumed run still prints the whole table.
rows = []
missing = []
for name in MODELS:
    for seed in SEEDS:
        result_path = result_path_of(name, seed)
        if not result_path.is_file():
            missing.append((name, seed, result_path))
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))["test"]
        row = {"model": name, "seed": seed}
        for family in ("classification", "pair", "sts", "retrieval"):
            for path, values in payload[family].items():
                row[KnowledgeDistiller._benchmark_name(path, "test")] = score_from_payload(
                    family, values
                )
        for key in SUMMARY_ORDER:
            value = payload["summary"].get(key)
            row[key] = np.nan if value is None else float(value)
        rows.append(row)

for name, seed, path in missing:
    print(f"[MISSING] {name} seed={seed}: {path}")
if not rows:
    raise SystemExit("No results.json to aggregate.")

by_seed = pd.DataFrame(rows)
metric_order = [name for name in BENCHMARK_ORDER + SUMMARY_ORDER if name in by_seed.columns]
by_seed = by_seed[["model", "seed", *metric_order]].sort_values(["model", "seed"])

grouped = by_seed.groupby("model", sort=False)[metric_order]
means, stds, counts = grouped.mean(), grouped.std(ddof=1), grouped.count()

wide = {}
for metric in metric_order:
    wide[f"{metric}_mean"] = means[metric]
    wide[f"{metric}_std"] = stds[metric]
    wide[f"{metric}_n"] = counts[metric]
summary = pd.DataFrame(wide)


def mean_pm_std(mean, std, digits=2):
    # Same convention as 00_main_results: one seed has no sample std to print.
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


display = pd.DataFrame(index=means.index)
for metric in metric_order:
    display[metric] = [
        mean_pm_std(mean * 100, std * 100) for mean, std in zip(means[metric], stds[metric])
    ]

by_seed_path = RUN_DIR / "by_seed.csv"
summary_path = RUN_DIR / "summary.csv"
by_seed.to_csv(by_seed_path, index=False)
summary.to_csv(summary_path)

print("\nBaselines — mean ± sample std over seeds, scores ×100")
with pd.option_context("display.width", 220, "display.max_columns", None):
    print(display.to_string())
print(f"\nPer-seed scores: {by_seed_path}")
print(f"Aggregated:      {summary_path}")
PY

trap - ERR INT TERM
echo
echo "Finished $(date -Is). Results under $RUN_DIR"
