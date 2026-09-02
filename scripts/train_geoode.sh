#!/bin/bash

echo "======================================"
echo "Training with GeoODE-KD method"
echo "======================================"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

METHOD="geoode"
TRAIN_DATA="../data/test_debug.csv"
STUDENT_MODEL="jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
LAMBDA_END=1.0
LAMBDA_CTR=0.5
SAVE_DIR="checkpoints/geoode"
# Instrumentation. Neither changes what is optimised -- a seeded run follows the
# same trajectory with both on -- so they are safe to leave on inside an ablation.
# DIAG_EVERY: per-term gradient norms, batch effective ranks, the signed H0
#   death-time residual and the student's own H1 diagram, every N steps.
# PROBE_EVERY: the structural ladder on a fixed probe of corpus sentences, into
#   probe_metrics.jsonl. 0 turns it off.
DIAG_EVERY=50
PROBE_EVERY=200
PROBE_SIZE=1024

python3 ../main.py \
    --method $METHOD \
    --train_data $TRAIN_DATA \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length $MAX_LENGTH \
    --lambda_end $LAMBDA_END \
    --lambda_ctr $LAMBDA_CTR \
    --diag_every $DIAG_EVERY \
    --probe_every $PROBE_EVERY \
    --probe_size $PROBE_SIZE \
    --save_dir $SAVE_DIR
