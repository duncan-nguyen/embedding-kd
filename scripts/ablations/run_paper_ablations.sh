#!/usr/bin/env bash
# Run every arm of docs/experiments.md: A1-A4 and the two sensitivities.
# 41 unique configurations x 3 seeds = 123 runs on Qwen3-0.6B -> MiniLM-L6-H384.
#
# The six tables overlap, so the plan is deduplicated: the recipe cell (PCA-384,
# student-selected gauge, L_end + L_H0) is a row of five tables and is trained once.
#
# The protocol is pinned here rather than left to the Python defaults, because these
# tables are only readable if every cell shares it: L_end at 1.0 and L_H0 at 0.5, no
# contrastive term, on the 14,760-row 15K corpus, configuration (c), seeds 42/43/44.
# A4 and S1 are the two tables that ablate and sweep the objective, so their cells
# override it -- that is what those tables are.
#
#     bash scripts/ablations/run_paper_ablations.sh --dry-run
#     bash scripts/ablations/run_paper_ablations.sh
#     GPUS="0 1" MAX_PARALLEL=2 bash scripts/ablations/run_paper_ablations.sh
#
# One table at a time, if the whole grid is too much at once:
#
#     bash scripts/ablations/run_paper_ablations.sh --tables a1
#     bash scripts/ablations/run_paper_ablations.sh --tables a2 a3
#
# Resume is free: a cell that already has a final test record is skipped, so
# re-running the same command after an interruption continues where it stopped, and
# running one table after another never retrains a shared cell. Read the tables back
# at any point, finished or not:
#
#     bash scripts/ablations/run_paper_ablations.sh --collect-only
#
# Start it under tmux or nohup if it will outlive the terminal:
#
#     tmux new -s ablations 'bash scripts/ablations/run_paper_ablations.sh'
#
# Timings: the runner records ms/step per cell, but a rate measured while other jobs
# shared the card is not a rate the efficiency table may print. Keep MAX_PARALLEL=1
# on a single GPU for anything whose ms/step is going to be published.
#
# Every argument is forwarded to the Python runner *after* the pinned flags, so
# anything given on the command line wins on a repeat: --tables, --seeds,
# --haar-draw, --keep-going, --retry-unfinished, --eval-retrieval.
#
# Environment:
#     GPUS                  devices, space separated (default: 0)
#     MAX_PARALLEL          jobs per GPU (default: 1)
#     PAIR                  teacher->student pair (default: qwen3_0.6b_to_minilm_h384)
#     TRAIN_DATA            corpus, relative to the repo root
#     RUN_ROOT              parent of the run directory (default: runs/)
#     RUN_NAME              resume an earlier sweep by name
#     CACHE_DIR             shared teacher cache (default: runs/teacher_cache)
#     PYTHON                interpreter (default: .venv/bin/python, else python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$REPO_ROOT/scripts/ablations/run_paper_ablations.py"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    python_bin="$REPO_ROOT/.venv/bin/python"
else
    python_bin="python3"
fi

PAIR="${PAIR:-qwen3_0.6b_to_minilm_h384}"
TRAIN_DATA="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
RUN_NAME="${RUN_NAME:-ablations_15k_${PAIR}}"
CACHE_DIR="${CACHE_DIR:-runs/teacher_cache}"
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Configuration (c): batch 128, five epochs, lr 7e-5, max_length 256 -- matched to
# run_decomposition.py and run_target_map_ablation.py so a number from this grid is
# comparable with a number from those.
command=(
    "$python_bin" "$RUNNER"
    --pair "$PAIR"
    --train-data "$TRAIN_DATA"
    --seeds 42 43 44
    --lambda-end 1.0
    --lambda-ctr 0.0
    --lambda-topo 0.5
    --batch-size 128
    --epochs 5
    --lr 7e-5
    --max-length 256
    --gauge-samples 16384
    --cache-dir "$CACHE_DIR"
    --run-root "$RUN_ROOT"
    --run-name "$RUN_NAME"
    --gpus $GPUS
    --max-parallel "$MAX_PARALLEL"
    "$@"
)

# --dry-run and --collect-only write no training output; they print a plan or read
# one back, so they get no log of their own and run in the foreground.
for argument in "$@"; do
    case "$argument" in
        --dry-run|--collect-only)
            cd "$REPO_ROOT"
            exec "${command[@]}"
            ;;
    esac
done

run_dir="$RUN_ROOT/$RUN_NAME"
mkdir -p "$run_dir"
log_path="$run_dir/sweep.log"

resume_hint() {
    echo
    echo "Sweep stopped. Every finished cell is kept; resume with the same command:"
    echo "    RUN_NAME=$RUN_NAME bash scripts/ablations/run_paper_ablations.sh"
    echo "Read back what is done so far:"
    echo "    RUN_NAME=$RUN_NAME bash scripts/ablations/run_paper_ablations.sh --collect-only"
    echo "Full log: $log_path"
}
trap resume_hint ERR INT TERM

echo "======================================"
echo "Paper ablations — A1-A4, S1, S2"
echo "  pair:    $PAIR"
echo "  corpus:  $TRAIN_DATA"
echo "  loss:    L_end(1.0) + L_H0(0.5), no contrastive term"
echo "           (A4 and S1 cells override the objective — that is what they ablate)"
echo "  seeds:   42 43 44"
echo "  gpus:    $GPUS x $MAX_PARALLEL job(s)"
echo "  run:     $run_dir"
echo "  log:     $log_path"
echo "  python:  $python_bin"
echo "  started  $(date -Is)"
echo "======================================"

cd "$REPO_ROOT"
# Appended, not truncated: a resume of the same run keeps the earlier passes.
"${command[@]}" 2>&1 | tee -a "$log_path"

trap - ERR INT TERM
echo
echo "Finished $(date -Is). Tables under $run_dir:"
echo "  table_a1_dimension_x_coordinate.csv  A1  reduction x coordinate, with energy"
echo "  table_a2_selection_signal.csv        A2  what the selection reads"
echo "  table_a3_representative.csv          A3  global vs local representative"
echo "  table_a4_components.csv              A4  endpoint x structural support"
echo "  sensitivity_s1_lambda_h0.csv         S1  lambda_H0 sweep"
echo "  sensitivity_s2_fit_set.csv           S2  fit-set size"
echo "  ablations_by_cell.csv                every cell, mean and std"
echo "  ablations_by_seed.csv                every run"
