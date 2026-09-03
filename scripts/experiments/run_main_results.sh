#!/usr/bin/env bash
# Run the whole main-results grid: all three teacher/student pairs, every
# baseline plus geoode, three seeds -- 63 runs.
#
# A thin front end for run_main_results.py, which is where the settings and the
# resume logic live. What it adds is the things a multi-day sweep wants and a
# bare python call does not: it picks the repo's own interpreter, names the run
# up front so the whole sweep is teed into one log next to its results, and
# prints the exact resume command if it stops early.
#
#     bash scripts/experiments/run_main_results.sh
#     bash scripts/experiments/run_main_results.sh --dry-run
#
# It runs in the foreground for hours, so start it under tmux or nohup:
#
#     tmux new -s sweep 'bash scripts/experiments/run_main_results.sh'
#     nohup bash scripts/experiments/run_main_results.sh > /dev/null 2>&1 &
#
# Every argument is forwarded to run_main_results.py, so its flags all work
# here: --pairs, --methods, --seeds, --dataset, --keep-going, --aggregate-only.
#
# Environment:
#     RUN_NAME              resume an earlier sweep by name (same as --run-name)
#     RUN_ROOT              parent of the run directory (default: runs/)
#     CUDA_VISIBLE_DEVICES  devices for each job (default: the script's own 0,1)
#     PYTHON                interpreter (default: .venv/bin/python, else python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$REPO_ROOT/scripts/experiments/run_main_results.py"

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

forwarded=("$@")
if [[ -z "$run_name" ]]; then
    run_name="${RUN_NAME:-main_results_all_pairs_$(date +%Y%m%d-%H%M%S)}"
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
    echo "    RUN_NAME=$run_name bash scripts/experiments/run_main_results.sh"
    echo "Full log: $log_path"
}
trap resume_hint ERR INT TERM

echo "======================================"
echo "Main results — all pairs, 3 seeds"
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
