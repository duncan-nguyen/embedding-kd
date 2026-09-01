#!/bin/bash

echo "======================================"
echo "Training with RKD method"
echo "======================================"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

METHOD="rkd"
TRAIN_DATA="../data/train_set/train_150k.csv"
STUDENT_MODEL="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
# Qwen3-Embedding is a decoder, so its sentence vector is the last token. An
# encoder teacher such as BGE-M3 reads "cls" instead. RKD applies this once, at
# cache time, so a wrong value here is baked into the cache rather than caught
# at the first step.
TEACHER_POOLING="last_token"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
W_DIST=25.0
W_ANGLE=50.0
# Re-encoding the corpus with the teacher is the most expensive thing in the
# pipeline and never changes between runs of the same pair, so the cache lives
# outside the run. The filename is derived from the teacher, the pooling and the
# corpus contents, so one directory holds every cache the project builds.
CACHE_DIR="../cache/teacher"
SAVE_DIR="checkpoints/rkd"

python3 ../main.py \
    --method $METHOD \
    --train_data $TRAIN_DATA \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --teacher_pooling $TEACHER_POOLING \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length $MAX_LENGTH \
    --w_dist $W_DIST \
    --w_angle $W_ANGLE \
    --save_dir $SAVE_DIR
