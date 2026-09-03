#!/usr/bin/env python3
"""Ablate the two factors of the frozen teacher target map ``P_T = P_PCA R``.

The recipe makes two independent claims and this grid is the control for each.

**Factor 1 -- the subspace** (``--subspace``). The claim is Eckart-Young: the
teacher's leading spectral subspace is what carries the signal. Its nulls are a
random subspace of the same rank, and the same map without the orthonormality:

    pca              centered PCA, mean removed before the SVD only (the paper's map)
    pca_full         centered PCA with the mean also removed when the map is applied
    svd              uncentered SVD -- the first direction may be the teacher mean
    random           Haar-random orthonormal columns: PCA's contract, no spectrum
    random_gaussian  Johnson-Lindenstrauss: the same random subspace, not an isometry
    mrl_prefix       the teacher's first d_S coordinates (Matryoshka-prefix interface)
    learned_t2s      a linear map d_T -> d_S trained with the student (EMO, sbert v5.5)
    learned_s2t      a linear map d_S -> d_T trained with the student (TALAS, LEAF)

The two learned arms are a different kind of control: they do not pick a subspace at
all, they let one be learned. They carry no gauge (a gauge orients a frozen basis and
a learned map has none), and their randomness is the initialisation of ``W``, so they
vary with ``--seeds`` rather than with ``--draws``.

**Factor 2 -- the orientation** (``--gauge``). The claim is Schoenemann: a gauge
kept aligned to the evolving student is better than the arbitrary gauge PCA returns.

    none             no rotation -- the coordinates the subspace fit produced
    procrustes       R fitted at init and refitted every epoch (the paper's R)
    random           a Haar-random rotation Q of identical cost -- THE control

``random`` is the sharp one. PCA's own basis is already an arbitrary gauge, so
``procrustes > none`` alone cannot tell "R is the *right* orientation" from "R is
*an* orientation": both stories predict it. ``procrustes > random`` can, and
``procrustes ~ random`` demotes the whole orientation factor to a null.

The random arms are stochastic, so a single cell is a single draw. ``--draws N``
runs N of them (different ``projection_seed`` / ``gauge_random_seed``); their spread
is the null band the spectral map and the fitted gauge have to clear, and reporting
a spectral-vs-random gap smaller than that band would be reporting noise.

Every cell shares one teacher cache and differs only in the target map, so the
comparison is clean by construction: same corpus, same objective, same schedule,
same student init. The objective is the method's default ``L_end + L_ctr``, passed
explicitly so the run log says what it was, and the corpus default is the one the
reported GeoODE rows used; benchmark contamination in that corpus inflates every
arm equally and so does not bias the comparison, though it does inflate the
absolute numbers.

Usage:
    # 1. see the plan (default: prints the commands, runs nothing)
    python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base

    # 2. run it
    python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base --execute

    # 3. read it back -- every planned cell, done or still missing
    python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base --collect

    # every subspace x gauge combination, three draws of each random arm
    python3 scripts/run_target_map_ablation.py --grid full --draws 3 --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The three teacher -> student pairs of the paper. Each carries the two settings
# that depend on the teacher's *family*: how its sentence vector is pooled (Qwen3
# is a decoder and reads the last token, BGE-M3 is an XLM-R encoder and reads CLS).
# Kept in sync with cell 1 of test_mdd.ipynb so a cell of this grid is directly
# comparable with the main-results run of the same pair.
PAIRS = {
    "qwen3_0.6b_to_minilm_h384": {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base",
        "teacher_pooling": "last_token",
    },
    "bge_m3_to_minilm_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large",
        "teacher_pooling": "cls",
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
        "teacher_pooling": "last_token",
    },
}

# Factor 1. Only the flags each arm actually decides are listed: the random arms
# leave pca_center_fit alone on purpose, because a map that never looks at the data
# has no mean to centre and naming the flag there would imply it matters.
SUBSPACE_ARMS = {
    "pca": {"projection_type": "pca", "pca_center_fit": True, "pca_subtract_mean": False},
    "pca_full": {"projection_type": "pca", "pca_center_fit": True, "pca_subtract_mean": True},
    "svd": {"projection_type": "pca", "pca_center_fit": False, "pca_subtract_mean": False},
    "random": {"projection_type": "random"},
    "random_gaussian": {"projection_type": "random_gaussian"},
    "mrl_prefix": {"projection_type": "mrl_prefix"},
    "learned_t2s": {"projection_type": "learned_t2s"},
    "learned_s2t": {"projection_type": "learned_s2t"},
}
# Redrawn per --draws: the map itself is sampled, so a single cell is a single draw.
STOCHASTIC_SUBSPACES = ("random", "random_gaussian")
# No frozen basis, therefore no gauge to put on it: these always run at gauge "none",
# and whatever randomness they have rides on the training seed.
LEARNED_SUBSPACES = ("learned_t2s", "learned_s2t")

# Factor 2.
GAUGE_ARMS = {
    "none": {"gauge_align": False},
    "procrustes": {"gauge_align": True, "gauge_rotation": "procrustes"},
    "random": {"gauge_align": True, "gauge_rotation": "random"},
}

# The cells this ablation was asked for: ours against the four things it has to beat.
# Read as one row per claim rather than as a grid --
#   pca/procrustes   ours: fixed subspace from the teacher's spectrum, with the
#                    Procrustes gauge refitted against the student every epoch
#   pca/none         the subspace alone (sentence-transformers <= v5.4, HPD's
#                    teacher side): is the gauge doing anything?
#   pca/random       the same subspace under a Haar rotation of identical cost: is
#                    the gauge doing anything *informative*?
#   random/none      no spectrum at all: is the subspace doing anything?
#   learned_*        the map learned instead of frozen: is adaptivity worth it?
REQUESTED_CELLS = [
    ("pca", "procrustes"),
    ("pca", "none"),
    ("pca", "random"),
    ("random", "none"),
    ("learned_t2s", "none"),
    ("learned_s2t", "none"),
]

SUMMARY_KEYS = ("avg_iod", "avg_ood", "avg_retrieval", "avg_all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pair", choices=sorted(PAIRS), default="qwen3_4b_to_bert_base")
    parser.add_argument(
        "--train_data",
        default="data/train_set/train_100k.csv",
        help="corpus, relative to the repo root; the default is the one the reported "
             "GeoODE rows were trained on",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output root (default runs/target_map_<pair>)",
    )
    parser.add_argument(
        "--grid",
        choices=["requested", "full"],
        default="requested",
        help="'requested' is ours against the four controls it has to beat (6 runs); "
             "'full' is every subspace x gauge combination (20 runs). Overridden by "
             "--subspace/--gauge",
    )
    parser.add_argument(
        "--subspace",
        nargs="+",
        choices=sorted(SUBSPACE_ARMS),
        default=None,
        help="subspace arms to cross with --gauge (overrides --grid)",
    )
    parser.add_argument(
        "--gauge",
        nargs="+",
        choices=sorted(GAUGE_ARMS),
        default=None,
        help="orientation arms to cross with --subspace (overrides --grid)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=1,
        help="draws of each stochastic arm (random subspace / random gauge). Their "
             "spread is the null band the fitted arms have to clear",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="training seeds; every cell is run at each of them",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lambda_ctr", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="skip ArguAna/FiQA/SCIDOCS. They are ~92k documents to encode per cell, "
             "so this is the switch that makes a large grid affordable -- at the cost "
             "of avg_retrieval and of avg_all meaning the same thing as elsewhere",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="shared directory for cached teacher embeddings (default "
             "<repo>/runs/teacher_cache). Every cell has the same teacher and "
             "corpus, so the teacher runs once for the whole grid -- and the "
             "directory is outside the run, so it is reused by later grids and by "
             "the notebook's method runs too",
    )
    parser.add_argument("--execute", action="store_true", help="run the plan")
    parser.add_argument("--collect", action="store_true", help="summarise the runs")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="value for CUDA_VISIBLE_DEVICES in the child processes",
    )
    parser.add_argument(
        "--stop_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop the grid at the first failing cell",
    )
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> list[dict]:
    """Enumerate the cells: (subspace, gauge) x draw x training seed."""
    if args.subspace or args.gauge:
        subspaces = args.subspace or sorted(SUBSPACE_ARMS)
        gauges = args.gauge or sorted(GAUGE_ARMS)
        cells = [(s, g) for s in subspaces for g in gauges]
    elif args.grid == "full":
        cells = [(s, g) for s in SUBSPACE_ARMS for g in GAUGE_ARMS]
    else:
        cells = list(REQUESTED_CELLS)
    # A learned map has no frozen basis, so its gauge column collapses to one cell
    # instead of silently running the same configuration three times.
    cells = [
        (subspace, "none" if subspace in LEARNED_SUBSPACES else gauge)
        for subspace, gauge in cells
    ]
    cells = list(dict.fromkeys(cells))
    if args.draws < 1:
        raise ValueError("--draws must be at least 1")

    plan = []
    for subspace, gauge in cells:
        random_subspace = subspace in STOCHASTIC_SUBSPACES
        random_gauge = gauge == "random"
        stochastic = random_subspace or random_gauge
        for draw in range(args.draws if stochastic else 1):
            settings = {**SUBSPACE_ARMS[subspace], **GAUGE_ARMS[gauge]}
            if random_subspace:
                settings["projection_seed"] = draw
            if random_gauge:
                settings["gauge_random_seed"] = draw
            for seed in args.seeds:
                name = f"{subspace}__{gauge}"
                if stochastic and args.draws > 1:
                    name += f"__d{draw}"
                if len(args.seeds) > 1:
                    name += f"__s{seed}"
                plan.append(
                    {
                        "name": name,
                        "subspace": subspace,
                        "gauge": gauge,
                        "draw": draw if stochastic else None,
                        "seed": seed,
                        "settings": settings,
                    }
                )
    return plan


def arm_flags(settings: dict) -> list[str]:
    """Turn an arm's settings into main.py flags.

    Booleans use argparse's --flag / --no-flag spelling, which is what makes an
    ablation readable in a log: --no-gauge_align says what was turned off.
    """
    flags = []
    for key, value in settings.items():
        if isinstance(value, bool):
            flags.append(f"--{key}" if value else f"--no-{key}")
        else:
            flags.extend([f"--{key}", str(value)])
    return flags


def output_root(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    return REPO_ROOT / "runs" / f"target_map_{args.pair}"


def build_command(args: argparse.Namespace, cell: dict) -> list[str]:
    pair = PAIRS[args.pair]
    root = output_root(args)
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "runs" / "teacher_cache"
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--method", "geoode",
        "--train_data", str(REPO_ROOT / args.train_data),
        "--student_model", pair["student"],
        "--teacher_model", pair["teacher"],
        "--teacher_pooling", pair["teacher_pooling"],
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--save_every", str(args.epochs),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length),
        "--lambda_ctr", str(args.lambda_ctr),
        "--seed", str(cell["seed"]),
        "--num_workers", str(args.num_workers),
        # No per-epoch evaluation: the grid is read off the final test row, and the
        # pair thresholds are swept there, so no extra pass is needed.
        "--eval_every", "0",
        "--pair_threshold_source", "test",
        "--cache_dir", str(cache_dir),
        "--save_dir", str(root / cell["name"]),
        "--no_wandb",
    ]
    if args.no_retrieval:
        command.append("--no_eval_retrieval")
    command.extend(arm_flags(cell["settings"]))
    # Epoch-wise refitting is the main Procrustes recipe. Every other gauge cell is
    # a fixed control and opts out explicitly now that refitting is the default.
    command.extend(
        ["--gauge_refit_every", "1"]
        if cell["gauge"] == "procrustes"
        else ["--gauge_refit_every", "0"]
    )
    return command


def final_test_record(save_dir: Path) -> dict | None:
    """The end-of-run record, identified by what it lacks: no "train" block."""
    metrics = save_dir / "metrics.jsonl"
    if not metrics.is_file():
        return None
    found = None
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("test") and not record.get("train"):
                found = record
    return found


def execute(args: argparse.Namespace, plan: list[dict]) -> list[dict]:
    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    status = []
    for position, cell in enumerate(plan, start=1):
        save_dir = root / cell["name"]
        if final_test_record(save_dir) is not None:
            print(f"[SKIP {position}/{len(plan)}] {cell['name']}: already has a final test")
            status.append({"name": cell["name"], "status": "skipped_complete", "seconds": 0.0})
            continue
        if (save_dir / "metrics.jsonl").exists():
            # Appending to a half-finished run would mix two configurations in one
            # metrics file, and the collector could not tell them apart.
            raise RuntimeError(
                f"{cell['name']} has an unfinished run at {save_dir}. Delete it or "
                "point --out somewhere else."
            )
        command = build_command(args, cell)
        log_path = root / f"{cell['name']}.log"
        print("\n" + "#" * 80)
        print(f"[{position}/{len(plan)}] {cell['name']}")
        print(shlex.join(command))
        print(f"log: {log_path}")
        print("#" * 80, flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                if "%|" not in line:
                    print(line, end="", flush=True)
            return_code = process.wait()
        elapsed = time.perf_counter() - started
        finished = return_code == 0 and final_test_record(save_dir) is not None
        status.append(
            {
                "name": cell["name"],
                "status": "complete" if finished else "failed",
                "seconds": elapsed,
            }
        )
        print(f"[{cell['name']}] {status[-1]['status']} in {elapsed / 60:.1f} min")
        if not finished and args.stop_on_error:
            raise RuntimeError(f"{cell['name']} failed; see {log_path}")
    return status


def read_projection(save_dir: Path) -> dict:
    """What the run recorded about the map it actually fitted."""
    path = save_dir / "teacher_projection.pt"
    if not path.is_file():
        return {}
    import torch  # local: the planner must not need a torch install

    saved = torch.load(path, map_location="cpu", weights_only=False)
    stats = saved.get("gauge_stats") or {}
    return {
        "projection_type": saved.get("projection_type"),
        "pca_center_fit": saved.get("pca_center_fit"),
        "pca_subtract_mean": saved.get("pca_subtract_mean"),
        "gauge_rotation": saved.get("gauge_rotation") if saved.get("gauge_align") else "none",
        "explained_energy": saved.get("explained_energy"),
        "cos_before": stats.get("cos_before"),
        "cos_after": stats.get("cos_after"),
        "cos_procrustes": stats.get("cos_procrustes"),
        # Only the interpolate arm sets this; True means this row's theta = 1 is not
        # the gauge the random arm with the same seed used.
        "endpoint_reflected": stats.get("endpoint_reflected"),
        "participation_ratio": stats.get("participation_ratio"),
    }


def collect(args: argparse.Namespace, plan: list[dict]) -> list[dict]:
    """One row per *planned* cell, so the table also says what has yet to be run."""
    root = output_root(args)
    rows = []
    for cell in plan:
        save_dir = root / cell["name"]
        row = {
            "name": cell["name"],
            "subspace": cell["subspace"],
            "gauge": cell["gauge"],
            "draw": cell["draw"],
            "seed": cell["seed"],
            "status": "missing",
        }
        record = final_test_record(save_dir)
        if record is not None:
            row["status"] = "done"
            summary = record["test"].get("summary", {})
            row.update(
                {key: (None if summary.get(key) is None else float(summary[key]))
                 for key in SUMMARY_KEYS}
            )
            row.update(read_projection(save_dir))
        rows.append(row)

    columns = [
        "name", "subspace", "gauge", "draw", "seed", "status",
        *SUMMARY_KEYS,
        "explained_energy", "cos_before", "cos_after", "cos_procrustes",
        "endpoint_reflected",
        "participation_ratio", "projection_type", "pca_center_fit",
        "pca_subtract_mean", "gauge_rotation",
    ]
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "target_map_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print_table(rows)
    done = sum(1 for row in rows if row["status"] == "done")
    print(f"\n{done}/{len(rows)} cells done. Written: {csv_path}")
    if done < len(rows):
        print("Run the rest with --execute (finished cells are skipped).")
    return rows


def print_table(rows: list[dict]) -> None:
    headers = ("cell", "status", "AVG(ALL)", "AVG(OOD)", "energy", "cos(z,tau)", "PR")

    def cell_values(row: dict) -> tuple[str, ...]:
        def pct(key):
            value = row.get(key)
            return "-" if value is None else f"{100.0 * value:.2f}"

        def num(key, fmt):
            value = row.get(key)
            return "-" if value is None else format(value, fmt)

        return (
            row["name"],
            row["status"],
            pct("avg_all"),
            pct("avg_ood"),
            "-" if row.get("explained_energy") is None else f"{100.0 * row['explained_energy']:.1f}%",
            num("cos_after", "+.3f"),
            num("participation_ratio", ".1f"),
        )

    table = [headers, *(cell_values(row) for row in rows)]
    widths = [max(len(line[index]) for line in table) for index in range(len(headers))]
    separator = "-+-".join("-" * width for width in widths)
    print()
    for index, line in enumerate(table):
        print(" | ".join(line[position].ljust(widths[position]) for position in range(len(line))))
        if index == 0:
            print(separator)


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    root = output_root(args)

    if args.collect:
        collect(args, plan)
        return

    print(f"pair:    {args.pair} ({PAIRS[args.pair]['teacher']} -> {PAIRS[args.pair]['student']})")
    print(f"corpus:  {args.train_data}")
    print(f"output:  {root}")
    print(f"cells:   {len(plan)}\n")
    for cell in plan:
        print(f"[{cell['name']}] {shlex.join(build_command(args, cell))}\n")

    if not args.execute:
        print("Dry run. Add --execute to run the grid, --collect to read it back.")
        return
    status = execute(args, plan)
    (root / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    collect(args, plan)


if __name__ == "__main__":
    main()
