#!/usr/bin/env bash
# Run the main-results grid on the 15k TALAS corpus: all three teacher/student
# pairs, every baseline plus geoode, three seeds -- 63 runs.
#
# Same front end as run_main_results.sh, and the same settings underneath; the
# only difference is the training corpus. This one pins
# --dataset talas_15k (data/train_set/merged_3_data_5k_each.csv, the 5k-each
# merge the TALAS paper trains on), which is what notebooks/00_main_results.ipynb
# uses, rather than the 100k corpus run_main_results.sh defaults to. Epochs,
# batch sizes, learning rates and the geoode extras are untouched, so the two
# sweeps differ in exactly one axis and their tables are comparable.
#
# The corpus is ~6.6x smaller than train_100k, so at the same 5 epochs this is a
# far shorter sweep than run_main_results.sh -- and it is the corpus without the
# SICK/STS-B train-test overlap of train_100k.
#
#     bash scripts/experiments/run_main_results_15k.sh
#     bash scripts/experiments/run_main_results_15k.sh --dry-run
#
# Start it under tmux or nohup if it is going to outlive the terminal:
#
#     tmux new -s sweep15k 'bash scripts/experiments/run_main_results_15k.sh'
#     nohup bash scripts/experiments/run_main_results_15k.sh > /dev/null 2>&1 &
#
# Every argument is forwarded to run_main_results.py, so its flags all work
# here: --pairs, --methods, --seeds, --keep-going, --aggregate-only. A --dataset
# of your own also still wins, since the pinned one is prepended.
#
# Environment:
#     RUN_NAME              resume an earlier sweep by name (same as --run-name)
#     RUN_ROOT              parent of the run directory (default: runs/)
#     CUDA_VISIBLE_DEVICES  devices for each job (default: the script's own 0,1)
#     PYTHON                interpreter (default: .venv/bin/python, else python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$REPO_ROOT/scripts/experiments/run_main_results.py"
DATASET="talas_15k"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    python_bin="$REPO_ROOT/.venv/bin/python"
else
    python_bin="python3"
fi

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"

# The run name has to be settled here rather than left to the Python default,
# because the log is written next to the results and this script needs to know
# that directory before the first job starts. A --run-name in the arguments wins:
# it is the resume path, and re-deriving a name would start a second sweep.
run_name=""
previous=""
for argument in "$@"; do
    case "$argument" in
        --run-name=*) run_name="${argument#--run-name=}" ;;
        *) [[ "$previous" == "--run-name" ]] && run_name="$argument" ;;
    esac
    previous="$argument"
done

forwarded=(--dataset "$DATASET" "$@")
if [[ -z "$run_name" ]]; then
    run_name="${RUN_NAME:-main_results_all_pairs_${DATASET}_$(date +%Y%m%d-%H%M%S)}"
    forwarded=(--run-name "$run_name" "${forwarded[@]}")
fi

command=("$python_bin" "$SWEEP" --run-root "$RUN_ROOT" "${forwarded[@]}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    command+=(--cuda-visible-devices "$CUDA_VISIBLE_DEVICES")
fi

# --dry-run prints a plan and writes nothing, so it gets no run directory and no
# log; anything else is a real sweep and is teed into its own results folder.
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        cd "$REPO_ROOT"
        exec "${command[@]}"
    fi
done

run_dir="$RUN_ROOT/$run_name"
mkdir -p "$run_dir"
log_path="$run_dir/sweep.log"

resume_hint() {
    echo
    echo "Sweep stopped. Everything finished so far is kept; resume with:"
    echo "    RUN_NAME=$run_name bash scripts/experiments/run_main_results_15k.sh"
    echo "Full log: $log_path"
}
trap resume_hint ERR INT TERM

echo "======================================"
echo "Main results — all pairs, 3 seeds, $DATASET"
echo "  run:    $run_dir"
echo "  log:    $log_path"
echo "  python: $python_bin"
echo "  started $(date -Is)"
echo "======================================"

cd "$REPO_ROOT"
# Appended, not truncated: a resume of the same run keeps the earlier passes.
"${command[@]}" 2>&1 | tee -a "$log_path"

trap - ERR INT TERM
echo
echo "Finished $(date -Is). Results and timings under $run_dir"
