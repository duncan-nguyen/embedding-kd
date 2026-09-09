#!/usr/bin/env python3
"""Every arm of docs/experiments.md in one grid: A1-A4 and the two sensitivities.

The six tables are cut from one plan because they overlap heavily -- the recipe cell
(PCA-384, student-selected gauge, L_end + L_H0) is a row of five of them, and the
teacher-frame and Haar cells are rows of four. A run directory is named after the
*configuration* rather than after the table asking for it, so a shared cell is
trained once and read into every table that needs it.

    A1  dimension reduction x coordinate selection   8 reductions x 3 coordinates
    A2  what signal the selection uses               teacher frame / Haar / shuffled
                                                     correspondence / unrelated
                                                     signal / matched student
    A3  global vs local representative               no selection / Haar / per
                                                     minibatch / global-initial /
                                                     global + one refresh / epoch-wise
    A4  component ablation                           endpoint target x structural
                                                     support, including the empty run
    S1  structural weight lambda_H0                  0, 0.1, 0.3, 0.5, 1.0
    S2  representative fit-set size                  2048, 4096, 8192, whole corpus

    41 unique cells x 3 seeds = 123 runs.

**The objective.** Every cell that is *not* about the objective runs the full recipe,
L_end + lambda_H0 * L_H0 with lambda_H0 = 0.5 and no contrastive term. A4 and S1 are
the tables that ablate and sweep the objective, so their cells override it -- that is
what those tables are. This is one protocol throughout, which the numbers currently
in experiments.md are not: those were collected across several sweeps, some endpoint
only and some full recipe, so expect this grid to move them.

**Two arms needed code that did not exist**: ``gauge_rotation="shuffled"`` and
``"unrelated"`` (A2 rows 3 and 4) run the same Procrustes solve against the student
with its rows permuted, and against an isotropic random cloud. See
src/teacher_projection.py::fit_gauge_rotation.

Usage:
    python3 scripts/ablations/run_paper_ablations.py --dry-run
    python3 scripts/ablations/run_paper_ablations.py --tables a1 a2
    python3 scripts/ablations/run_paper_ablations.py --collect-only
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

from src import job_runner  # noqa: E402

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

# --------------------------------------------------------------- factor 1: reduction
# Only what an arm actually decides is listed. The random arms leave pca_center_fit
# alone on purpose: a map that never looks at the teacher has no mean to centre.
# ``projection_rank`` 0 (omitted) is the maximal feasible rank, min(d_T, d_S); the
# narrower ranks keep the student architecture fixed and narrow the *interface*.
REDUCTIONS = {
    "pca_64": ("PCA-64", {"projection_type": "pca", "projection_rank": 64}),
    "pca_128": ("PCA-128", {"projection_type": "pca", "projection_rank": 128}),
    "pca_256": ("PCA-256", {"projection_type": "pca", "projection_rank": 256}),
    "pca_384": ("PCA-384", {"projection_type": "pca"}),
    "random_1": ("Random subspace #1", {"projection_type": "random", "projection_seed": 0}),
    "random_2": ("Random subspace #2", {"projection_type": "random", "projection_seed": 1}),
    "random_3": ("Random subspace #3", {"projection_type": "random", "projection_seed": 2}),
    "svd_uncentered": ("Uncentered SVD", {"projection_type": "pca", "pca_center_fit": False}),
    # Not in the current experiments.md -- the previous draft's Main Table 1 had them.
    # Kept addressable through --tables learned rather than deleted.
    "learned_t2s": ("Learned T->S projector", {"projection_type": "learned_t2s"}),
    "learned_s2t": ("Learned S->T projector", {"projection_type": "learned_s2t"}),
}

# ------------------------------------------------------------ factor 2: coordinates
# ``per_batch`` is not a gauge at all: --endpoint_loss procrustes re-solves the
# rotation in closed form on every batch and throws it away, which makes the endpoint
# term invariant to the coordinates rather than committing to any. ``refresh`` is
# resolved against --epochs so that exactly one refit fires, near the middle of the
# run: the distiller refits after epoch e when (e+1) % N == 0 and e+1 < epochs.
COORDINATES = {
    "as_reduced": ("Teacher frame / no selection", {"gauge_align": False, "gauge_refit_every": 0}),
    "haar": ("Haar random", {
        "gauge_align": True, "gauge_rotation": "random",
        "gauge_random_seed": 0, "gauge_refit_every": 0,
    }),
    "shuffled": ("Shuffled student correspondence", {
        "gauge_align": True, "gauge_rotation": "shuffled",
        "gauge_random_seed": 0, "gauge_refit_every": 0,
    }),
    "unrelated": ("Random / unrelated student signal", {
        "gauge_align": True, "gauge_rotation": "unrelated",
        "gauge_random_seed": 0, "gauge_refit_every": 0,
    }),
    "per_batch": ("Per-minibatch selection", {
        "gauge_align": False, "gauge_refit_every": 0, "endpoint_loss": "procrustes",
    }),
    "selected_init": ("Global selection, initial only", {
        "gauge_align": True, "gauge_rotation": "procrustes", "gauge_refit_every": 0,
    }),
    "selected_refresh": ("Global selection + one refresh", {
        "gauge_align": True, "gauge_rotation": "procrustes", "gauge_refit_every": "__ONE_REFRESH__",
    }),
    "selected": ("Global epoch-wise selection", {
        "gauge_align": True, "gauge_rotation": "procrustes", "gauge_refit_every": 1,
    }),
    # A learned map has no frozen basis to orient, so its coordinate column collapses.
    "implicit": ("Implicit (learned jointly)", {"gauge_align": False, "gauge_refit_every": 0}),
}
LEARNED_REDUCTIONS = ("learned_t2s", "learned_s2t")
A1_REDUCTIONS = (
    "pca_64", "pca_128", "pca_256", "pca_384",
    "random_1", "random_2", "random_3", "svd_uncentered",
)
A1_COORDINATES = ("as_reduced", "haar", "selected")

# ------------------------------------------------------- cells that vary the objective
# A4 and S1 are the tables about the objective, so these are the only cells that touch
# it. Everything above runs the full recipe. ``lambda_topo`` and ``lambda_gram`` given
# as None mean "the protocol's value"; a number overrides it.
OBJECTIVE_CELLS = {
    "no_teacher": ("None", "None", {
        "reduction": "pca_384", "coordinate": "as_reduced",
        "lambda_end": 0.0, "lambda_topo": 0.0,
    }),
    "endpoint_teacher_frame": ("Teacher-frame endpoint", "None", {
        "reduction": "pca_384", "coordinate": "as_reduced", "lambda_topo": 0.0,
    }),
    "endpoint_selected": ("Student-selected endpoint", "None", {
        "reduction": "pca_384", "coordinate": "selected", "lambda_topo": 0.0,
    }),
    "h0_only": ("None", "H_0", {
        "reduction": "pca_384", "coordinate": "as_reduced", "lambda_end": 0.0,
    }),
    "selected_gram": ("Student-selected endpoint", "Gram / invariant structural loss", {
        "reduction": "pca_384", "coordinate": "selected",
        "lambda_topo": 0.0, "lambda_gram": "__PROTOCOL_STRUCTURAL__",
    }),
    "selected_knn": ("Student-selected endpoint", "NN-distance structural loss", {
        "reduction": "pca_384", "coordinate": "selected",
        "structural_loss": "knn_distribution",
    }),
}

S1_WEIGHTS = (0.0, 0.1, 0.3, 0.5, 1.0)
S2_SIZES = (2048, 4096, 8192, None)  # None = the whole corpus (the protocol's value)


def _cell_name(reduction: str, coordinate: str) -> str:
    return f"{reduction}__{coordinate}"


def s1_cell(weight: float, protocol_topo: float) -> str:
    """S1 shares two of its five rows with cells the other tables already define."""
    if weight == 0.0:
        return "endpoint_selected"
    if weight == protocol_topo:
        return _cell_name("pca_384", "selected")
    return f"pca_384__selected__topo{weight:g}"


def s2_cell(size: int | None) -> str:
    """S2's largest row is the protocol: the fit set is capped at the corpus length."""
    return _cell_name("pca_384", "selected") if size is None else f"pca_384__selected__fit{size}"


def build_cells(args: argparse.Namespace) -> dict[str, dict]:
    """Every cell any requested table needs, keyed by name and defined exactly once."""
    cells: dict[str, dict] = {}

    def add(name: str, reduction: str, coordinate: str, **overrides) -> None:
        existing = cells.get(name)
        entry = {
            "name": name, "reduction": reduction, "coordinate": coordinate,
            "overrides": overrides,
        }
        # A name is a configuration: if two tables ask for the same name they must
        # mean the same run, or one of them would silently read the other's number.
        assert existing is None or existing == entry, f"conflicting definition of {name}"
        cells[name] = entry

    wanted = set(args.tables)
    if "all" in wanted:
        wanted = {"a1", "a2", "a3", "a4", "s1", "s2"}

    if "a1" in wanted:
        for reduction in A1_REDUCTIONS:
            for coordinate in A1_COORDINATES:
                add(_cell_name(reduction, coordinate), reduction, coordinate)
    if "a2" in wanted:
        for coordinate in ("as_reduced", "haar", "shuffled", "unrelated", "selected"):
            add(_cell_name("pca_384", coordinate), "pca_384", coordinate)
    if "a3" in wanted:
        for coordinate in (
            "as_reduced", "haar", "per_batch", "selected_init", "selected_refresh", "selected"
        ):
            add(_cell_name("pca_384", coordinate), "pca_384", coordinate)
    if "a4" in wanted:
        for name, (_, _, spec) in OBJECTIVE_CELLS.items():
            spec = dict(spec)
            add(name, spec.pop("reduction"), spec.pop("coordinate"), **spec)
        add(_cell_name("pca_384", "selected"), "pca_384", "selected")
    if "s1" in wanted:
        for weight in S1_WEIGHTS:
            name = s1_cell(weight, args.lambda_topo)
            if name == "endpoint_selected":
                add(name, "pca_384", "selected", lambda_topo=0.0)
            elif name == _cell_name("pca_384", "selected"):
                add(name, "pca_384", "selected")
            else:
                add(name, "pca_384", "selected", lambda_topo=weight)
    if "s2" in wanted:
        for size in S2_SIZES:
            name = s2_cell(size)
            if size is None:
                add(name, "pca_384", "selected")
            else:
                add(name, "pca_384", "selected", gauge_align_samples=size)
    if "learned" in wanted:
        for reduction in LEARNED_REDUCTIONS:
            add(_cell_name(reduction, "implicit"), reduction, "implicit")
    if not cells:
        raise ValueError("no table selected")
    return cells


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pair", choices=sorted(PAIRS), default="qwen3_0.6b_to_minilm_h384")
    parser.add_argument(
        "--tables", nargs="+", default=["all"],
        choices=["all", "a1", "a2", "a3", "a4", "s1", "s2", "learned"],
        help="'all' is A1-A4 + S1 + S2 (41 cells). 'learned' adds the two "
             "learned-projector arms, which the current experiments.md no longer has",
    )
    parser.add_argument("--train-data", default="data/train_set/merged_3_data_5k_each.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--lambda-end", type=float, default=1.0)
    parser.add_argument("--lambda-ctr", type=float, default=0.0)
    parser.add_argument(
        "--lambda-topo", type=float, default=0.5,
        help="the protocol's lambda_H0, and the weight the matched structural "
             "controls of A4 are given",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--h0-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--gauge-samples", type=int, default=16384,
        help="fit-set size for the gauge; capped at the corpus length, which is what "
             "makes S2's largest row the protocol cell",
    )
    parser.add_argument(
        "--haar-draw", choices=["per_seed", "shared"], default="per_seed",
        help="'per_seed' redraws the Haar/shuffled/unrelated controls for each "
             "training seed, so their error bars contain the spread over draws",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--cache-dir", default="runs/teacher_cache")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--eval-retrieval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--retry-unfinished", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args(argv)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def output_root(args: argparse.Namespace) -> Path:
    return resolve_path(args.run_root) / (args.run_name or f"ablations_15k_{args.pair}")


def one_refresh_every(epochs: int) -> int:
    """The refit stride that fires exactly once, near the middle of the run.

    The distiller refits after epoch ``e`` when ``(e + 1) % N == 0`` and
    ``e + 1 < epochs``. ``N = (epochs + 1) // 2`` leaves one multiple of ``N`` strictly
    below ``epochs``: five epochs refit after the third and nowhere else.
    """
    if epochs < 2:
        raise ValueError("one refresh needs at least two epochs")
    return (epochs + 1) // 2


def flag_list(flags: dict) -> list[str]:
    """argparse's --flag / --no-flag spelling, so a log says what was turned off."""
    out: list[str] = []
    for key, value in flags.items():
        if isinstance(value, bool):
            out.append(f"--{key}" if value else f"--no-{key}")
        else:
            out.extend([f"--{key}", str(value)])
    return out


def cell_flags(args: argparse.Namespace, cell: dict, seed: int) -> dict:
    """The protocol, then the cell's reduction, coordinates and objective overrides."""
    flags = {
        "lambda_end": args.lambda_end,
        "lambda_ctr": args.lambda_ctr,
        "lambda_topo": args.lambda_topo,
        "lambda_gram": 0.0,
        "structural_loss": "h0",
        "gauge_align_samples": args.gauge_samples,
    }
    flags.update(REDUCTIONS[cell["reduction"]][1])
    coordinate = cell["coordinate"]
    flags.update(COORDINATES[coordinate][1])
    if flags.get("gauge_refit_every") == "__ONE_REFRESH__":
        flags["gauge_refit_every"] = one_refresh_every(args.epochs)
    if "gauge_random_seed" in flags and args.haar_draw == "per_seed":
        # A random gauge is a draw, not a setting: holding one draw fixed across the
        # seeds would report a particular rotation and call it arbitrary.
        flags["gauge_random_seed"] = args.seeds.index(seed)
    for key, value in cell["overrides"].items():
        flags[key] = args.lambda_topo if value == "__PROTOCOL_STRUCTURAL__" else value
    # A learned map has no frozen basis, so no gauge and no rank apply to it.
    if cell["reduction"] in LEARNED_REDUCTIONS:
        flags.pop("projection_rank", None)
    return flags


def build_command(args: argparse.Namespace, cell: dict, seed: int, save_dir: Path) -> list[str]:
    pair = PAIRS[args.pair]
    command = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--method", "geoode",
        "--train_data", str(resolve_path(args.train_data)),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--student_pooling", "cls",
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--save_every", str(args.epochs),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length),
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        # The grid is read off the final test row and the pair thresholds are swept
        # there, so no per-epoch evaluation pass is needed.
        "--eval_every", "0",
        "--pair_threshold_source", "test",
        "--cache_dir", str(resolve_path(args.cache_dir)),
        "--save_dir", str(save_dir),
        # L_H0 is read off the teacher's own d_T cache, so it never passes through
        # P_T and cannot confound the interface factor.
        "--topo_batch_size", str(args.h0_batch_size or args.batch_size),
        "--topo_metric", "chord",
        "--topo_teacher_source", "original",
        "--no_wandb",
        *flag_list(cell_flags(args, cell, seed)),
    ]
    if not args.eval_retrieval:
        command.append("--no_eval_retrieval")
    return command


def build_jobs(args: argparse.Namespace, cells: dict[str, dict]) -> list[dict]:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be non-empty and unique")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    root = output_root(args)
    jobs = []
    for name, cell in cells.items():
        for seed in args.seeds:
            run_dir = root / name / f"seed_{seed}"
            jobs.append({
                **cell,
                "name": f"{name}/seed_{seed}",
                "cell_name": name,
                "seed": seed,
                "run_dir": run_dir,
                "log_path": run_dir / "train.log",
                "command": build_command(args, cell, seed, run_dir),
            })
    return jobs


# ------------------------------------------------------------------------ reading back

SUMMARY_KEYS = ("avg_iod", "avg_ood", "avg_all")
PROJECTION_KEYS = (
    "retained_energy", "cos_before", "cos_after", "cos_procrustes",
    "participation_ratio", "gauge_samples",
)


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
    """The end-of-run record, identified by what it lacks: no "train" block."""
    return _last_json_record(
        directory / "metrics.jsonl",
        lambda record: bool(record.get("test")) and not record.get("train"),
    )


def read_projection(directory: Path) -> dict:
    """What the run recorded about the map and the gauge it actually fitted."""
    path = directory / "teacher_projection.pt"
    if not path.is_file():
        return {}
    import torch  # local: planning and collecting must not need a torch install

    saved = torch.load(path, map_location="cpu", weights_only=False)
    stats = saved.get("gauge_stats") or {}
    return {
        "retained_energy": saved.get("explained_energy"),
        "cos_before": stats.get("cos_before"),
        "cos_after": stats.get("cos_after"),
        "cos_procrustes": stats.get("cos_procrustes"),
        "participation_ratio": stats.get("participation_ratio"),
        # The fit set is capped at the corpus length, so this is what S2 reports
        # rather than what was requested.
        "gauge_samples": stats.get("samples"),
    }


def median_ms_per_step(directory: Path, skip: int = 20) -> float | None:
    """Median step time in ms, first ``skip`` steps dropped as warmup."""
    path = directory / "step_metrics.jsonl"
    if not path.is_file():
        return None
    seconds = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line).get("step_seconds")
                if value:
                    seconds.append(float(value))
    seconds = seconds[skip:]
    return 1000.0 * statistics.median(seconds) if seconds else None


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def _pct(value, digits: int = 2) -> str:
    return "-" if value is None else f"{100.0 * float(value):.{digits}f}"


def _num(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(float(value), spec)


def collect(args: argparse.Namespace, jobs: list[dict], cells: dict[str, dict]) -> None:
    root = output_root(args)
    co_located = len(job_runner.gpu_slots(args.gpus, args.max_parallel)) > 1

    by_seed = []
    for job in jobs:
        final = final_test_record(job["run_dir"])
        row = {
            "cell": job["cell_name"],
            "reduction": job["reduction"],
            "coordinate": job["coordinate"],
            "seed": job["seed"],
            "status": "done" if final else "missing",
            "ms_per_step": median_ms_per_step(job["run_dir"]),
            "timing_co_located": co_located,
            "run_dir": str(job["run_dir"]),
        }
        if final:
            summary = final["test"].get("summary", {})
            row.update({key: summary.get(key) for key in SUMMARY_KEYS})
            row.update(read_projection(job["run_dir"]))
        by_seed.append(row)

    by_cell = []
    for name, cell in cells.items():
        done = [r for r in by_seed if r["cell"] == name and r["status"] == "done"]
        out = {
            "cell": name,
            "reduction_label": REDUCTIONS[cell["reduction"]][0],
            "coordinate_label": COORDINATES[cell["coordinate"]][0],
            "overrides": ";".join(f"{k}={v}" for k, v in cell["overrides"].items()),
            "n": len(done),
        }
        for key in (*SUMMARY_KEYS, "ms_per_step"):
            out[f"{key}_mean"], out[f"{key}_std"] = _mean_std(
                [float(r[key]) for r in done if r.get(key) is not None]
            )
        for key in PROJECTION_KEYS:
            values = [r[key] for r in done if r.get(key) is not None]
            out[key] = float(values[0]) if values else None
        out["timing_co_located"] = co_located
        by_cell.append(out)
    index = {row["cell"]: row for row in by_cell}

    _write_csv(
        root / "ablations_by_seed.csv", by_seed,
        ["cell", "reduction", "coordinate", "seed", "status", *SUMMARY_KEYS,
         *PROJECTION_KEYS, "ms_per_step", "timing_co_located", "run_dir"],
    )
    _write_csv(
        root / "ablations_by_cell.csv", by_cell,
        ["cell", "reduction_label", "coordinate_label", "overrides", "n",
         *[f"{k}_{s}" for k in (*SUMMARY_KEYS, "ms_per_step") for s in ("mean", "std")],
         *PROJECTION_KEYS, "timing_co_located"],
    )

    wanted = {"a1", "a2", "a3", "a4", "s1", "s2"} if "all" in args.tables else set(args.tables)
    if "a1" in wanted:
        _emit_a1(root, index)
    if "a2" in wanted:
        _emit_simple(
            root, index, "table_a2_selection_signal.csv",
            "A2 -- what information coordinate selection uses",
            "Selection signal",
            [(COORDINATES[c][0], _cell_name("pca_384", c))
             for c in ("as_reduced", "haar", "shuffled", "unrelated", "selected")],
        )
    if "a3" in wanted:
        _emit_simple(
            root, index, "table_a3_representative.csv",
            "A3 -- global vs local representative selection",
            "Representative strategy",
            [(COORDINATES[c][0], _cell_name("pca_384", c))
             for c in ("as_reduced", "haar", "per_batch",
                       "selected_init", "selected_refresh", "selected")],
        )
    if "a4" in wanted:
        _emit_a4(root, index)
    if "s1" in wanted:
        _emit_simple(
            root, index, "sensitivity_s1_lambda_h0.csv",
            "S1 -- structural weight lambda_H0", "lambda_H0",
            [(f"{w:g}", s1_cell(w, args.lambda_topo)) for w in S1_WEIGHTS],
            energy=False,
        )
    if "s2" in wanted:
        _emit_simple(
            root, index, "sensitivity_s2_fit_set.csv",
            "S2 -- representative fit-set size", "Fit-set size",
            [("whole corpus" if s is None else f"{s:,}", s2_cell(s)) for s in S2_SIZES],
            energy=False, fit_size=True,
        )

    done = sum(row["status"] == "done" for row in by_seed)
    print(f"\n{done}/{len(by_seed)} runs done -> {root}")
    if done < len(by_seed):
        print("Re-run without --collect-only to finish the rest; completed runs are skipped.")


def _render(rows: list[tuple[str, ...]]) -> None:
    if len(rows) < 2:
        print("  (nothing collected yet)")
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for position, row in enumerate(rows):
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
        if position == 0:
            print("-+-".join("-" * width for width in widths))


def _emit_a1(root: Path, index: dict) -> None:
    rows = []
    for reduction in A1_REDUCTIONS:
        row = {"reduction": REDUCTIONS[reduction][0]}
        energies = [
            index[_cell_name(reduction, c)]["retained_energy"]
            for c in A1_COORDINATES
            if _cell_name(reduction, c) in index
            and index[_cell_name(reduction, c)]["retained_energy"] is not None
        ]
        row["retained_energy"] = energies[0] if energies else None
        for coordinate in A1_COORDINATES:
            cell = index.get(_cell_name(reduction, coordinate), {})
            row[f"{coordinate}_mean"] = cell.get("avg_all_mean")
            row[f"{coordinate}_std"] = cell.get("avg_all_std")
            row[f"{coordinate}_n"] = cell.get("n", 0)
        rows.append(row)
    _write_csv(
        root / "table_a1_dimension_x_coordinate.csv", rows,
        ["reduction", "retained_energy",
         *[f"{c}_{s}" for c in A1_COORDINATES for s in ("mean", "std", "n")]],
    )
    print("\nA1 -- dimension reduction x coordinate selection (AVG ALL over seeds)")
    _render([
        ("Dimension reduction", "Energy", "As reduced", "Haar", "Student-selected"),
        *[(r["reduction"], _num(r["retained_energy"]),
           *(_pct(r[f"{c}_mean"]) for c in A1_COORDINATES)) for r in rows],
    ])


def _emit_simple(
    root: Path, index: dict, filename: str, title: str, label: str,
    entries: list[tuple[str, str]], *, energy: bool = True, fit_size: bool = False,
) -> None:
    """One row per entry: a label, optionally the retained energy, and the average."""
    rows = []
    for text, name in entries:
        cell = index.get(name, {})
        row = {label: text, "cell": name, "n": cell.get("n", 0),
               "avg_all_mean": cell.get("avg_all_mean"),
               "avg_all_std": cell.get("avg_all_std")}
        if energy:
            row["retained_energy"] = cell.get("retained_energy")
        if fit_size:
            row["gauge_samples"] = cell.get("gauge_samples")
        rows.append(row)
    columns = [label, *(["retained_energy"] if energy else []),
               *(["gauge_samples"] if fit_size else []),
               "avg_all_mean", "avg_all_std", "n", "cell"]
    _write_csv(root / filename, rows, columns)
    print(f"\n{title}")
    header = (label, *(("Energy",) if energy else ()),
              *(("Fitted rows",) if fit_size else ()), "Avg.", "std", "n")
    _render([header, *[
        (r[label], *((_num(r["retained_energy"]),) if energy else ()),
         *((_num(r.get("gauge_samples"), ".0f"),) if fit_size else ()),
         _pct(r["avg_all_mean"]), _pct(r["avg_all_std"]), str(r["n"])) for r in rows
    ]])


def _emit_a4(root: Path, index: dict) -> None:
    entries = [
        *[(endpoint, structural, name)
          for name, (endpoint, structural, _) in OBJECTIVE_CELLS.items()],
        ("Student-selected endpoint", "H_0", _cell_name("pca_384", "selected")),
    ]
    rows = []
    for endpoint, structural, name in entries:
        cell = index.get(name, {})
        rows.append({
            "endpoint_target": endpoint, "structural_support": structural,
            "retained_energy": cell.get("retained_energy"),
            "avg_all_mean": cell.get("avg_all_mean"),
            "avg_all_std": cell.get("avg_all_std"),
            "n": cell.get("n", 0), "cell": name,
        })
    _write_csv(
        root / "table_a4_components.csv", rows,
        ["endpoint_target", "structural_support", "retained_energy",
         "avg_all_mean", "avg_all_std", "n", "cell"],
    )
    print("\nA4 -- component ablation of GATE-KD")
    _render([
        ("Endpoint target", "Structural support", "Energy", "Avg.", "std", "n"),
        *[(r["endpoint_target"], r["structural_support"], _num(r["retained_energy"]),
           _pct(r["avg_all_mean"]), _pct(r["avg_all_std"]), str(r["n"])) for r in rows],
    ])


# ---------------------------------------------------------------------------- running


def prepare_jobs(args: argparse.Namespace, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    pending, status = [], []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for job in jobs:
        directory = job["run_dir"]
        if final_test_record(directory):
            status.append({"name": job["name"], "status": "skipped_complete"})
            continue
        if (directory / "metrics.jsonl").exists():
            # Appending to a half-finished run would mix two configurations in one
            # metrics file and the collector could not tell them apart.
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
    """Embed the corpus once, before any job starts: every cell shares the teacher."""
    pair = PAIRS[args.pair]
    command = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--method", "geoode",
        "--train_data", str(resolve_path(args.train_data)),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--max_length", str(args.max_length),
        "--cache_dir", str(resolve_path(args.cache_dir)),
        "--cache_only", "--no_eval_retrieval", "--no_wandb",
    ]
    print(f"[cache] {shlex.join(command)}")
    subprocess.run(
        command, cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpus[0])}, check=True,
    )


def execute(args: argparse.Namespace, jobs: list[dict]) -> list[dict]:
    pending, status = prepare_jobs(args, jobs)
    if not pending:
        return status
    slots = job_runner.gpu_slots(args.gpus, jobs_per_gpu=args.max_parallel)
    prewarm_cache(args)

    def on_finish(job: dict, row: dict) -> dict:
        complete = row["returncode"] == 0 and final_test_record(job["run_dir"])
        return {**row, "status": "complete" if complete else "failed"}

    rows = job_runner.run_jobs_parallel(
        pending, cwd=REPO_ROOT,
        env={**os.environ, "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"},
        slots=slots, poll_seconds=args.poll_seconds,
        stop_on_error=not args.keep_going, on_finish=on_finish,
    )
    return [*status, *rows]


def main() -> None:
    args = parse_args()
    train_data = resolve_path(args.train_data)
    if not train_data.is_file():
        raise FileNotFoundError(f"Training corpus not found: {train_data}")
    cells = build_cells(args)
    jobs = build_jobs(args, cells)
    root = output_root(args)

    print(f"pair:    {args.pair} ({PAIRS[args.pair]['teacher']} -> {PAIRS[args.pair]['student']})")
    print(f"corpus:  {train_data}")
    print(f"loss:    L_end({args.lambda_end}) + L_H0({args.lambda_topo}), "
          f"lambda_ctr={args.lambda_ctr} (A4 and S1 cells override it)")
    print(f"seeds:   {args.seeds}")
    print(f"tables:  {' '.join(args.tables)}")
    print(f"output:  {root}")
    print(f"plan:    {len(cells)} cells x {len(args.seeds)} seeds = {len(jobs)} runs")
    if args.dry_run:
        for job in jobs:
            print(f"[{job['name']}] {shlex.join(job['command'])}")
        print("\nDry run: nothing was written.")
        return
    if args.collect_only:
        collect(args, jobs, cells)
        return

    root.mkdir(parents=True, exist_ok=True)
    status = execute(args, jobs)
    _write_csv(root / "run_status.csv", status,
               ["name", "status", "gpu", "returncode", "seconds"])
    collect(args, jobs, cells)


if __name__ == "__main__":
    main()
