from .base_config import BaseConfig


class GeoODEConfig(BaseConfig):
    """GATE-KD: fixed PCA subspace with epoch-wise gauge alignment."""

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
    # Weight of the topological term L_topo = L_H0 + lambda_h1 * L_H1. It constrains
    # the *shape* of the cloud, not the position of any point, and is read off the
    # teacher's own d_T-dimensional cache rather than the projected target, so it is
    # the one term that does not depend on P_T. 0 is the recipe; > 0 is the "+ topo"
    # row. L_H0 is the squared error between the sorted finite H0 death times (= MST
    # edge weights) of the student and teacher batches; death times live in [0, 2]
    # (chord), so the term is small -- sweep the weight over decades, 0.01-1.0.
    # CLI: --lambda_topo.
    lambda_topo = 0.0
    # Weight lambda_1 of the H1 half of L_topo: W_2^2 between the teacher's and the
    # student's 1-dimensional persistence diagrams, i.e. the cycles of the batch
    # rather than its merge tree, with low-persistence cycles matched to the diagonal.
    # 0 leaves L_topo the pure H0 term. Needs the optional `gudhi` package, and builds
    # the batch's full 2-skeleton (O(B^3) simplices) on both sides of the loss, so it
    # is the expensive row of the ablation and its cost is set by batch_size, not by
    # d: ~7 ms/step at batch_size=32, ~430 ms at 128, ~6 s at 256. W_2^2 is a *sum*
    # over matched cycles while L_H0 is a mean over B - 1 death times, so this weight
    # carries the scale ratio as well: start an order of magnitude below lambda_topo.
    # CLI: --lambda_h1.
    lambda_h1 = 0.0
    # Size of the point cloud the topological terms read, in rows. 0 -- the default
    # -- makes that cloud the training batch, one diagram per step, which is what it
    # always was. Any b >= 2 splits the batch into B // b disjoint clouds of b rows,
    # averages their losses and drops the B mod b trailing rows from L_topo only. The
    # two sizes answer different questions -- batch_size sets the variance of the
    # gradient, this sets the scale at which the filtration reads the geometry, since
    # L_H0 compares exactly b - 1 death times -- so this is the knob for sweeping the
    # topological scale with the optimizer held fixed. It also makes L_H1 cheaper by
    # (B/b)^2, the 2-skeleton being O(b^3) per chunk. CLI: --topo_batch_size.
    topo_batch_size = 0
    # Ground metric of both persistence diagrams on the unit sphere: "chord" is the
    # Euclidean distance sqrt(2 - 2cos), "angular" the geodesic acos(cos), "cosine"
    # the (non-metric) 1 - cos. CLI: --topo_metric.
    topo_metric = "chord"
    # "original" bypasses P_T; "projected" reads the d_S endpoint targets.
    # CLI: --topo_teacher_source.
    topo_teacher_source = "original"

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
    # Active rank of a fixed teacher interface before it is embedded in the
    # d_S-wide student space. 0 keeps the paper recipe (the maximal feasible
    # rank, min(d_T, d_S)). Smaller values support the controlled width-bottleneck
    # sweep without changing the student architecture; the unused coordinates
    # are exactly zero before the gauge rotation. CLI: --projection_rank.
    projection_rank = 0
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
    # R: orthogonal Procrustes alignment of the PCA coordinates to the student.
    # The calibration subset is selected once before training and reused unchanged
    # by every refit. It contains up to this many corpus sentences (must be >> d_S).
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
    # minimisation over O(d_S)); 1 is the paper recipe and 0 is the fixed-gauge
    # ablation. The final epoch is not followed by a redundant refit.
    gauge_refit_every = 1

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
