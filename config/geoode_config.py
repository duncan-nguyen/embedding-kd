from .base_config import BaseConfig


class GeoODEConfig(BaseConfig):
    """GeoODE-KD: teacher-guided geometric dynamics distillation."""

    distill_method = "geoode"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    student_special_token = "##"
    teacher_special_token = "G"

    # Semantic potential, Eq. (20). The paper fixes alpha = 1 and tunes beta.
    alpha = 1.0
    beta = 1.0

    # Objective weights, Eq. (38). lambda_end and lambda_dyn are fixed at 1 and
    # lambda_ctr is the one tuned on a validation split.
    lambda_end = 1.0
    lambda_dyn = 1.0
    lambda_ctr = 0.1
    contrastive_temperature = 0.05

    # Depth-dependent guidance s(t), Eq. (28): "linear" (t), "power" (t^p), "constant".
    guidance_schedule = "linear"
    guidance_power = 1.0

    # Pool(.) of Eq. (7) applied at every depth. "cls" matches the pooling the
    # evaluation code uses, so the supervised endpoint is the embedding that is scored.
    student_pooling = "cls"
    include_embedding_layer = False
    stop_grad_target = True

    # Sample the per-depth diagnostics every N optimizer steps (0 disables them).
    # They are what the paper's hypotheses are stated over, and at this cadence the
    # extra cost is one parameter-free pass over the layers per ~50 steps.
    depth_log_every = 50

    # Second view for the contrastive term, Eq. (37): "dropout" runs the same text
    # twice under independent dropout masks (the paper's choice); "pair" reuses the
    # second sentence of the training pair instead.
    contrastive_view = "dropout"

    eps_norm = 1e-12

    # Fixed teacher dimensionality reduction P_T = P_PCA R, Eq. (8).
    pca_center_fit = True
    pca_subtract_mean = False
    # R: orthogonal Procrustes alignment of the PCA coordinates to the untrained
    # student (closed form, fitted once). Removes the arbitrary gauge of the PCA
    # basis from the endpoint loss; off is the ablation. The alignment is fitted on
    # up to this many corpus sentences (must be >> d_S).
    gauge_align = True
    gauge_align_samples = 16384
    # Re-estimate R every N epochs against the current student (alternating exact
    # minimisation over O(d_S)); 0 keeps the initial gauge for the whole run.
    gauge_refit_every = 0

    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    cache_teacher = True
    cache_path = "cache/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "checkpoints/geoode"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
