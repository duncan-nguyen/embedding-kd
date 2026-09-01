from .base_config import BaseConfig


class RKDConfig(BaseConfig):
    """RKD: relational knowledge distillation on cached teacher embeddings."""

    distill_method = "rkd"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    student_special_token = "##"
    teacher_special_token = "Ġ"

    # Park et al. (2019), Sec. 4: the two relational potentials are added to the
    # student's own loss with lambda_RKD-D = 25 and lambda_RKD-A = 50.
    w_task = 1.0
    w_dist = 25.0
    w_angle = 50.0
    huber_delta = 1.0

    # The teacher cache is L2-normalised and every benchmark scores cosine
    # similarity, so the student's relations are measured on the same sphere.
    # --no-normalize_student is the raw-Euclidean ablation.
    normalize_student = True

    # Pooling of the student sentence vector. "cls" is what the evaluation code
    # reads, so the vector being supervised is the vector being scored.
    student_pooling = "cls"

    # Temperature of the in-batch contrastive task loss.
    temperature = 0.1
    # Floor under the square root of the pairwise distances. Larger than the
    # 1e-12 the other methods use because this one has to stay a floor after a
    # cast to half precision, where 1e-12 is exactly zero -- see RelationalKD.
    eps_norm = 1e-6

    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    cache_teacher = True
    cache_path = "cache/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"


    save_dir = "checkpoints/rkd"
