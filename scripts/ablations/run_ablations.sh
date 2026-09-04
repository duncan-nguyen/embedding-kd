#!/usr/bin/env bash
# Run the complete 15K ablation suite:
#   1. target-map/interface ablation: 6 arms x 3 seeds = 18 jobs
#   2. endpoint/topology decomposition: 6 arms x 3 seeds = 18 jobs
#
# Usage:
#   bash scripts/ablations/run_ablations.sh --dry-run
#   bash scripts/ablations/run_ablations.sh
#   bash scripts/ablations/run_ablations.sh --target-map-only
#   bash scripts/ablations/run_ablations.sh --decomposition-only
#   bash scripts/ablations/run_ablations.sh --collect-only
#
# Environment overrides:
#   PAIR=qwen3_0.6b_to_minilm_h384
#   TRAIN_DATA=data/train_set/merged_3_data_5k_each.csv
#   SEEDS="42 43 44"
#   GPUS="0 1"
#   MAX_PARALLEL=1
#   NUM_WORKERS=16
#   RUN_ROOT=runs
#   CACHE_DIR=runs/teacher_cache
#   PYTHON=.venv/bin/python

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_MAP="$REPO_ROOT/scripts/ablations/run_target_map_ablation.py"
DECOMPOSITION="$REPO_ROOT/scripts/ablations/run_decomposition.py"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    python_bin="$REPO_ROOT/.venv/bin/python"
else
    python_bin="python3"
fi

pair="${PAIR:-qwen3_0.6b_to_minilm_h384}"
train_data="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"
seed_text="${SEEDS:-42 43 44}"
gpu_text="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
max_parallel="${MAX_PARALLEL:-1}"
num_workers="${NUM_WORKERS:-16}"
run_root="${RUN_ROOT:-$REPO_ROOT/runs}"
cache_dir="${CACHE_DIR:-$REPO_ROOT/runs/teacher_cache}"

read -r -a seeds <<< "$seed_text"
gpu_text="${gpu_text//,/ }"
read -r -a gpus <<< "$gpu_text"
gpu_csv="$(IFS=,; echo "${gpus[*]}")"

run_target_map=true
run_decomposition=true
mode="execute"

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

for argument in "$@"; do
    case "$argument" in
        --dry-run) mode="dry-run" ;;
        --collect-only) mode="collect" ;;
        --target-map-only) run_decomposition=false ;;
        --decomposition-only) run_target_map=false ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; usage >&2; exit 2 ;;
    esac
done

target_command=(
    "$python_bin" "$TARGET_MAP"
    --pair "$pair"
    --train_data "$train_data"
    --seeds "${seeds[@]}"
    --num_workers "$num_workers"
    --cache_dir "$cache_dir"
    --out "$run_root/target_map_15k_$pair"
    --cuda_visible_devices "$gpu_csv"
)
decomposition_command=(
    "$python_bin" "$DECOMPOSITION"
    --pair "$pair"
    --train-data "$train_data"
    --seeds "${seeds[@]}"
    --num-workers "$num_workers"
    --cache-dir "$cache_dir"
    --run-root "$run_root"
    --run-name "decomposition_15k_$pair"
    --gpus "${gpus[@]}"
    --max-parallel "$max_parallel"
)

case "$mode" in
    dry-run)
        target_command+=(--dry-run)
        decomposition_command+=(--dry-run)
        ;;
    collect)
        target_command+=(--collect)
        decomposition_command+=(--collect-only)
        ;;
    execute)
        target_command+=(--execute)
        ;;
esac

cd "$REPO_ROOT"
echo "Ablation suite: pair=$pair, seeds=${seeds[*]}, data=$train_data"

if $run_target_map; then
    echo
    echo "=== Target-map/interface ablation ==="
    "${target_command[@]}"
fi

if $run_decomposition; then
    echo
    echo "=== Endpoint/topology decomposition ==="
    "${decomposition_command[@]}"
fi

echo
echo "Ablation suite finished. Results are under $run_root."
