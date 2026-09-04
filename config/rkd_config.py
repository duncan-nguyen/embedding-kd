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
    w_dist = 25.0
    w_angle = 50.0
    huber_delta = 1.0

    # Park et al. put the student's own loss in at weight 1. Lowered here, and
    # this is the one place this row departs from the published recipe.
    #
    # Both potentials are scale-free -- psi_D divides by the batch mean, psi_A is
    # a cosine -- so the gradient they deliver grows as the batch they are
    # measured on shrinks. A [CLS] head starts nearly collapsed, so RKD owns
    # essentially the whole update at step 0 and roughly a tenth of it once the
    # student has spread out; watch student_spread against teacher_spread in the
    # step metrics. The consequence is a term that converges slowly at the tail,
    # and under the 5-epoch budget shared with the other rows the run ends well
    # short of what the objective can reach. At that budget w_task = 0.1 was at
    # least as good as 1.0 at every learning rate probed, and much better at the
    # low ones; 1.0 restores the published recipe.
    w_task = 0.1

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
    # 2e-5 is the untuned shared default; GATE-KD, the method this row is the
    # baseline for, is run at 7e-5 and Stella at 5e-5. Since RKD is the slowest
    # of the objectives to converge, leaving it at the lowest rate in the table
    # is what handicaps it, so it gets the same peak rate as GATE-KD. min_lr is
    # deliberately left at 2e-6, which is the floor GATE-KD actually runs with
    # once the notebook overrides its rate: same peak, same schedule shape.
    learning_rate = 7e-5
    min_lr = 2e-6

    cache_teacher = True
    cache_path = "cache/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "checkpoints/rkd"
