#!/usr/bin/env bash
# Shared setup for every scripts/methods/<method>/train.sh. Sourced, never run.
#
# It resolves the repo root from its own location and runs there, so a method
# script works from any working directory -- `bash scripts/methods/talas/train.sh`
# from the repo root behaves exactly like calling it from inside its own folder.
#
# What lives here is only what every method shares: the pair, the corpus, the
# teacher cache and the two environment variables the run reads. Everything a
# method alone decides -- its loss weights, its batch size, its epochs -- is left
# to config/<method>_config.py. Every CLI flag of main.py defaults to None and a
# None never overrides the config, so a value omitted below is the config's, not
# a silent default. Restating one here would only let the two drift apart.
#
# Every value is overridable from the environment, no file edited:
#
#     TRAIN_DATA=data/train_set/train_200k.csv bash scripts/methods/talas/train.sh
#     TEACHER_MODEL=Qwen/Qwen3-Embedding-4B STUDENT_MODEL=google-bert/bert-base-uncased \
#         bash scripts/methods/geoode/train.sh
#
# and any argument given to a method script is forwarded to main.py after the
# flags the script itself sets, so it wins on a repeat:
#
#     bash scripts/methods/geoode/train.sh --lambda_topo 0.1 --no_wandb

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Training is single-process. Two visible CUDA devices place the student on
# cuda:0 and the teacher on cuda:1; one device puts both on cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# The pair the main tables are built on. TEACHER_POOLING is the teacher's
# sentence vector: Qwen3-Embedding is a decoder, so it is the last token; an
# encoder teacher such as BGE-M3 reads "cls". The cached methods apply it once,
# at cache time, so a wrong value here is baked into the cache rather than
# caught at the first step.
STUDENT_MODEL="${STUDENT_MODEL:-nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TEACHER_POOLING="${TEACHER_POOLING:-last_token}"

# Sub-word marker the token-level alignments (cdm/emo) strip before comparing
# token strings: "Ġ" for the byte-level BPE of Qwen3, "▁" for a SentencePiece
# teacher such as BGE-M3. Empty leaves it to the config, which is the Qwen3
# value -- set it when switching teacher family, or cdm and emo align against a
# marker their teacher never emits.
TEACHER_SPECIAL_TOKEN="${TEACHER_SPECIAL_TOKEN:-}"

TRAIN_DATA="${TRAIN_DATA:-data/train_set/train_150k.csv}"

# Re-encoding the corpus with the teacher is the most expensive thing in the
# pipeline and never changes between runs of the same pair, so the cache lives
# outside the run. The filename is derived from the teacher, the pooling and the
# corpus contents, so one directory holds every cache the project builds and a
# run either finds exactly its own or misses. Read by the cached methods
# (talas/geoode/rkd) only; the online ones ignore it.
CACHE_DIR="${CACHE_DIR:-cache/teacher}"

# Under checkpoints/, which is gitignored. SAVE_DIR is per method, so it is set
# by run_distill rather than here.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"

# run_distill <method> [extra main.py flags...]
#
# Passes the shared flags and then the caller's. The teacher flags are omitted
# when TEACHER_MODEL is empty, which is how the simcse control -- the one run
# that loads no teacher -- reuses this function.
run_distill() {
    local method="$1"
    shift

    echo "======================================"
    echo "Training with ${method} method"
    echo "  student: ${STUDENT_MODEL}"
    echo "  teacher: ${TEACHER_MODEL:-<none>}"
    echo "  corpus:  ${TRAIN_DATA}"
    echo "======================================"

    local -a args=(
        --method "$method"
        --train_data "$TRAIN_DATA"
        --student_model "$STUDENT_MODEL"
        --save_dir "${CHECKPOINT_ROOT}/${method}"
    )
    if [[ -n "$TEACHER_MODEL" ]]; then
        args+=(--teacher_model "$TEACHER_MODEL" --teacher_pooling "$TEACHER_POOLING")
    fi
    if [[ -n "$TEACHER_SPECIAL_TOKEN" ]]; then
        args+=(--teacher_special_token "$TEACHER_SPECIAL_TOKEN")
    fi

    cd "$REPO_ROOT"
    python3 main.py "${args[@]}" "$@"
}
