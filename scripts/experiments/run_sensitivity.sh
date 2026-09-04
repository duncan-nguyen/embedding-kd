#!/usr/bin/env bash
# Run the 15K GATE-KD sensitivity grid: 10 OFAT arms x 3 seeds = 30 jobs.
#
#   bash scripts/experiments/run_sensitivity.sh --dry-run
#   bash scripts/experiments/run_sensitivity.sh
#   RUN_NAME=sensitivity_15k_qwen3_0.6b_to_minilm_h384 \
#       bash scripts/experiments/run_sensitivity.sh       # resume
#
# Every argument is forwarded to run_sensitivity.py. Useful overrides:
#   --gpus 0 1 --max-parallel 2
#   --sweeps lambda_topo batch_size
#   --collect-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$REPO_ROOT/scripts/experiments/run_sensitivity.py"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    python_bin="$REPO_ROOT/.venv/bin/python"
else
    python_bin="python3"
fi

run_root="${RUN_ROOT:-$REPO_ROOT/runs}"
run_name=""
pair="qwen3_0.6b_to_minilm_h384"
previous=""
for argument in "$@"; do
    case "$argument" in
        --run-name=*) run_name="${argument#--run-name=}" ;;
        --run-root=*) run_root="${argument#--run-root=}" ;;
        --pair=*) pair="${argument#--pair=}" ;;
        *) [[ "$previous" == "--run-name" ]] && run_name="$argument" ;;
    esac
    [[ "$previous" == "--run-root" ]] && run_root="$argument"
    [[ "$previous" == "--pair" ]] && pair="$argument"
    previous="$argument"
done
run_name="${run_name:-${RUN_NAME:-sensitivity_15k_${pair}}}"

command=("$python_bin" "$SWEEP" --run-root "$run_root" --run-name "$run_name" "$@")
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" || "$argument" == "--collect-only" ]]; then
        cd "$REPO_ROOT"
        exec "${command[@]}"
    fi
done

run_dir="$run_root/$run_name"
mkdir -p "$run_dir"
log_path="$run_dir/sweep.log"

resume_hint() {
    echo
    echo "Sweep stopped. Completed final-test jobs are reusable; resume with:"
    echo "    RUN_NAME=$run_name bash scripts/experiments/run_sensitivity.sh"
    echo "Full sweep log: $log_path"
}
trap resume_hint ERR INT TERM

cd "$REPO_ROOT"
"${command[@]}" 2>&1 | tee -a "$log_path"

trap - ERR INT TERM
echo "Finished. Results: $run_dir"
