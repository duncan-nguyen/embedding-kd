from .base_config import BaseConfig


class GeoODEConfig(BaseConfig):
    """GeoODE-KD: endpoint distillation against a frozen teacher map (L_end + L_ctr)."""

    distill_method = "geoode"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    student_special_token = "##"
    teacher_special_token = "G"

    # Objective weights, Eq. (38). The objective is L_end + L_ctr: the endpoint term
    # at weight 1 and the contrastive regulariser at the weight tuned on a
    # validation split. Nothing else is in it.
    lambda_end = 1.0
    lambda_ctr = 0.5
    contrastive_temperature = 0.05
    # Form of the endpoint term. "cosine" is the recipe (Eq. 36: 1 - <z, tau> on
    # the unit sphere). "mse" is the sentence-transformers <= v5.4 distillation
    # recipe used as a baseline: squared error between the *unnormalised* pooled
    # student state and the projected teacher target, which is then *not*
    # re-normalised after P_T (so the target keeps the norm the projection left it
    # with). Pair it with --lambda_ctr 0 and --no-gauge_align to reproduce that
    # recipe exactly (PCA target + MSE, nothing else). CLI: --endpoint_loss.
    endpoint_loss = "cosine"
    # Control for Prop. 3: weight of a pairwise-similarity (Gram) term between the
    # student's and the target's batch Gram matrices. 0 is the recipe; > 0 is the
    # "+ Gram" row of the recipe ablation, expected to be redundant with L_end once
    # the interface is a fixed orthonormal map. CLI: --lambda_gram.
    lambda_gram = 0.0

    # Pool(.) of Eq. (7) applied at every depth. "cls" matches the pooling the
    # evaluation code uses, so the supervised endpoint is the embedding that is scored.
    student_pooling = "cls"
    include_embedding_layer = False

    # Second view for the contrastive term, Eq. (37): "dropout" runs the same text
    # twice under independent dropout masks (the paper's choice); "pair" reuses the
    # second sentence of the training pair instead.
    contrastive_view = "dropout"

    eps_norm = 1e-12

    # Fixed teacher dimensionality reduction P_T = P_PCA R, Eq. (8).
    # --- factor 1: which d_S-dimensional subspace of the teacher to keep ---
    # "pca" is the paper's map. "random" (Haar-random orthonormal columns) and
    # "random_gaussian" (Johnson-Lindenstrauss, no orthonormality) are the
    # data-independent controls for the Eckart-Young claim. "learned_t2s" and
    # "learned_s2t" are the *adaptive* baselines: a linear map trained with the
    # student instead of fitted and frozen, mapping the teacher down or the student
    # up respectively. They add parameters during training only -- the deployed
    # model is still the plain student encoder. CLI: --projection_type /
    # --projection_seed.
    projection_type = "pca"
    projection_seed = 0
    # Learning rate of the learned target map, as a multiple of the student's. The
    # learned arms are baselines the recipe is measured against, so this exists to
    # let them be tuned rather than strawmanned; 1.0 gives them the student's own
    # schedule, which is what the methods they stand for do.
    learned_projector_lr_scale = 1.0
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
    # Position on the geodesic between the Procrustes gauge (0) and the random one
    # (1); read only by gauge_rotation = "interpolate". CLI: --gauge_theta.
    gauge_theta = 0.5
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
