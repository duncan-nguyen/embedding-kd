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

    # Objective weights, Eq. (38). The default objective is L_end + L_ctr: the
    # endpoint term at weight 1 and the contrastive regulariser at the weight tuned
    # on a validation split. The per-transition velocity term is off by default
    # (--lambda_vel > 0 turns it on) -- it is an ablation, not part of the recipe.
    lambda_end = 1.0
    lambda_vel = 0.0
    # Weak semantic-descent constraint, Eq. (33): the deep half of the transitions
    # is penalised only when it *raises* E_sem. Off by default so the objective
    # stays the one the reported runs used; --lambda_desc turns it on.
    lambda_desc = 0.0
    lambda_ctr = 0.5
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

    # Gram matrix the relational energy E_geo is measured against, Eq. (18):
    # "native" uses the teacher's own cosine matrix (no projection loss),
    # "projected" the Gram of the P_T-projected targets (ablation).
    relational_target = "native"

    # Fixed teacher dimensionality reduction P_T = P_PCA R, Eq. (8).
    # --- factor 1: which d_S-dimensional subspace of the teacher to keep ---
    # "pca" is the paper's map; "random" (Haar-random orthonormal columns) and
    # "random_gaussian" (Johnson-Lindenstrauss, no orthonormality) are the
    # data-independent controls for the Eckart-Young claim. CLI:
    # --projection_type / --projection_seed.
    projection_type = "pca"
    projection_seed = 0
    # Centre the cache before the SVD: True is centered PCA, False the uncentered
    # SVD ablation, in which the teacher's mean vector is allowed to be the first
    # retained direction. Ignored by the random arms, which never look at the data.
    pca_center_fit = True
    # Whether the mean is also removed when the map is *applied*; the fit above only
    # decides which directions are picked. True is the textbook PCA transform.
    pca_subtract_mean = False
    # --- factor 2: the orientation inside that subspace ---
    # R: orthogonal Procrustes alignment of the PCA coordinates to the untrained
    # student (closed form, fitted once). Removes the arbitrary gauge of the PCA
    # basis from the endpoint loss; off is the ablation. The alignment is fitted on
    # up to this many corpus sentences (must be >> d_S).
    gauge_align = True
    gauge_align_samples = 16384
    # Which rotation gauge_align applies: "procrustes" is the informative one,
    # "random" a Haar-random rotation of the same cost. PCA alone is already an
    # arbitrary gauge, so "random" is the control that says whether Procrustes wins
    # by *being the right* orientation rather than by rotating at all.
    gauge_rotation = "procrustes"
    gauge_random_seed = 0
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
