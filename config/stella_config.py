from .base_config import BaseConfig


class StellaConfig(BaseConfig):
    distill_method = "stella"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    output_dim1 = 1024
    output_dim2 = 512
    output_dim3 = 256
    output_dim4 = 128
    pooling = "cls"

    epochs_stage1 = 2
    epochs_stage2 = 3

    w_cos_stage1 = 0.5
    w_sim_stage1 = 2.0
    w_tri_stage1 = 0.5

    w_task = 0.5
    w_cos_stage2 = 0.5
    w_sim_stage2 = 2.0
    w_tri_stage2 = 0.5

    temperature = 0.1

    batch_size = 32
    learning_rate = 5e-5

    # Both stage losses L2-normalise the teacher embedding before reading it and
    # never touch its token states, so the teacher can be run once, offline, and
    # read from the same normalised cache TALAS/GATE-KD/RKD read. Pass --cache_dir
    # to share one cache between runs of the same pair; cache_path is the
    # single-file fallback for a run that names its own.
    cache_teacher = True
    cache_path = "cache/teacher_train.pt"
    normalize_cache = True
    cache_dtype = "float32"

    train_data_path = "data/train.csv"
    save_dir = "checkpoints/stella"
    save_every = 1
