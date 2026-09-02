import argparse

from config import (
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    GeoODEConfig,
    RKDConfig,
    SimCSEConfig,
    StellaConfig,
    TALASConfig,
)
from distiller import KnowledgeDistiller


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Embeddings Model"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="cdm",
        choices=["cdm", "dskd", "emo", "stella", "talas", "geoode", "rkd", "simcse"],
        help="Distillation method to use",
    )

    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to training data CSV file"
    )
    parser.add_argument(
        "--eval_data", type=str, default=None, help="Path to evaluation data CSV file"
    )

    parser.add_argument(
        "--student_model", type=str, default=None, help="Student model name or path"
    )
    parser.add_argument(
        "--teacher_model", type=str, default=None, help="Teacher model name or path"
    )
    parser.add_argument(
        "--teacher_pooling",
        choices=["last_token", "mean", "cls"],
        default=None,
        help="Pooling of the teacher sentence vector: applied once at cache time by "
        "talas/geoode/rkd and every step by cdm/dskd/emo/stella. Qwen3-Embedding "
        "reads last_token, encoder teachers such as BGE-M3 read cls",
    )
    parser.add_argument(
        "--teacher_special_token",
        type=str,
        default=None,
        help="Sub-word marker of the teacher tokenizer that the CDM token alignment "
        'strips before comparing token strings ("Ġ" for Qwen3 byte-level BPE, '
        '"▁" for SentencePiece teachers such as BGE-M3). EMO reads it as the '
        'teacher BOS token string instead ("<s>")',
    )
    parser.add_argument(
        "--student_special_token",
        type=str,
        default=None,
        help='Sub-word marker of the student tokenizer ("##" for WordPiece students; '
        'EMO reads it as the student CLS token string, "[CLS]")',
    )

    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Save a periodic checkpoint every N epochs (must be positive)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )

    parser.add_argument("--w_task", type=float, default=None, help="Task loss weight")
    parser.add_argument(
        "--alpha_dtw", type=float, default=None, help="DTW KD loss weight"
    )
    parser.add_argument(
        "--lambda_end",
        type=float,
        default=None,
        help="GeoODE-KD: weight of the endpoint distillation loss",
    )
    parser.add_argument(
        "--lambda_ctr",
        type=float,
        default=None,
        help="GeoODE-KD: weight of the contrastive regularizer",
    )
    parser.add_argument(
        "--endpoint_loss",
        choices=["cosine", "mse", "procrustes"],
        default=None,
        help='GeoODE-KD: form of the endpoint term. "cosine" is the recipe; "mse" is '
        "the sentence-transformers <= v5.4 baseline (squared error between the "
        "unnormalised student state and the projected, un-renormalised teacher "
        "target; per-sample sum over dimensions, batch mean, so it is on the "
        "scale of the cosine term). Combine with --lambda_ctr 0 --no-gauge_align "
        "for the pure PCA + MSE recipe. "
        '"procrustes" re-solves an orthogonal alignment of the batch every step '
        "(per-step re-alignment control)",
    )
    parser.add_argument(
        "--lambda_gram",
        type=float,
        default=None,
        help="GeoODE-KD: weight of a pairwise-similarity (Gram) term between student "
        'and target batch Gram matrices. 0 is the recipe; > 0 is the "+ Gram" '
        "control of the recipe ablation",
    )
    parser.add_argument(
        "--lambda_topo",
        type=float,
        default=None,
        help="GeoODE-KD: weight of the topological term L_topo = L_H0 + "
        "lambda_h1 * L_H1, comparing the student batch's persistence diagrams to "
        'those of the *unprojected* teacher batch. 0 is the recipe; > 0 is the "+ '
        'topo" control. Death times are O(1), so sweep the weight over decades',
    )
    parser.add_argument(
        "--lambda_h1",
        type=float,
        default=None,
        help="GeoODE-KD: weight lambda_1 of the H1 half of L_topo -- W_2^2 between "
        "the teacher's and the student's 1-dimensional persistence diagrams, "
        "low-persistence cycles matched to the diagonal. 0 leaves L_topo the pure "
        "H0 term. Requires the optional 'gudhi' package and costs O(B^3) simplices "
        "per batch on both sides",
    )
    parser.add_argument(
        "--topo_metric",
        choices=["chord", "angular", "cosine"],
        default=None,
        help="GeoODE-KD: ground metric of the H0 diagram on the unit sphere -- "
        '"chord" (Euclidean), "angular" (geodesic) or "cosine" (1 - cos)',
    )
    parser.add_argument(
        "--projection_type",
        choices=[
            "pca",
            "random",
            "random_gaussian",
            "mrl_prefix",
            "learned_t2s",
            "learned_s2t",
        ],
        default=None,
        help='GeoODE-KD: how the teacher targets reach the student dimension. "pca" '
        'is the paper\'s frozen spectral map; "random" draws a Haar-random '
        'orthonormal subspace and "random_gaussian" an unnormalised '
        "Johnson-Lindenstrauss map -- the two data-independent controls for the "
        'Eckart-Young claim; "mrl_prefix" keeps the teacher\'s leading '
        'coordinates (the Matryoshka-prefix interface); "learned_t2s" and '
        '"learned_s2t" replace the frozen map with a linear layer trained '
        "alongside the student (teacher mapped down, or student mapped up into "
        "the teacher space) -- the adaptive baselines",
    )
    parser.add_argument(
        "--projection_seed",
        type=int,
        default=None,
        help="GeoODE-KD: draw index of the random teacher projection. Different "
        "seeds are different draws of the same control, so their spread is the "
        "null band the PCA map has to clear",
    )
    parser.add_argument(
        "--learned_projector_lr_scale",
        type=float,
        default=None,
        help="GeoODE-KD: learning rate of a learned target map (--projection_type "
        "learned_*) as a multiple of the student's. The learned arms are the "
        "baselines the frozen map is measured against, so they get a sweep "
        "(e.g. 1 and 5) rather than a single untuned setting",
    )
    parser.add_argument(
        "--pca_center_fit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GeoODE-KD: centre the cache before the SVD that picks the directions. "
        "--no-pca_center_fit is the uncentered-SVD ablation, in which the "
        "teacher mean vector may itself be the first retained direction",
    )
    parser.add_argument(
        "--pca_subtract_mean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GeoODE-KD: subtract the corpus mean before applying P_T (textbook PCA, "
        "removes the common component of the teacher embeddings)",
    )
    parser.add_argument(
        "--gauge_align",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GeoODE-KD: Procrustes-align the PCA target coordinates to the untrained "
        "student (P_T = P_PCA R). --no-gauge_align is the ablation",
    )
    parser.add_argument(
        "--gauge_rotation",
        choices=["procrustes", "random", "interpolate", "rank_one"],
        default=None,
        help='GeoODE-KD: which rotation --gauge_align applies. "procrustes" is the '
        'informative gauge fitted to the student init; "random" is a '
        "Haar-random rotation of identical cost, the control that separates "
        '"the right orientation" from "an orientation"; "interpolate" is the '
        "geodesic point --gauge_theta of the way from the Procrustes gauge to the "
        'random one; "rank_one" is the Householder map aligning only the two '
        "mean directions",
    )
    parser.add_argument(
        "--gauge_theta",
        type=float,
        default=None,
        help="GeoODE-KD: theta in [0, 1] for --gauge_rotation interpolate "
        "(0 = Procrustes, 1 = random)",
    )
    parser.add_argument(
        "--gauge_random_seed",
        type=int,
        default=None,
        help="GeoODE-KD: draw index of the random gauge rotation Q",
    )
    parser.add_argument(
        "--gauge_refit_every",
        type=int,
        default=None,
        help="GeoODE-KD: re-estimate the gauge R against the current student every N "
        "epochs (0 = keep the initial gauge)",
    )
    parser.add_argument(
        "--w_dist",
        type=float,
        default=None,
        help="RKD: weight of the distance-wise relational loss (lambda_RKD-D)",
    )
    parser.add_argument(
        "--w_angle",
        type=float,
        default=None,
        help="RKD: weight of the angle-wise relational loss (lambda_RKD-A)",
    )
    parser.add_argument(
        "--normalize_student",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="RKD: L2-normalise the student embeddings before measuring relations, "
        "so they are compared on the same sphere as the normalised teacher "
        "cache. --no-normalize_student is the raw-Euclidean ablation",
    )
    parser.add_argument(
        "--simcse_view",
        choices=["dropout", "pair"],
        default=None,
        help='SimCSE-only: positive view. "dropout" encodes the same sentence twice '
        '(unsupervised SimCSE); "pair" uses the paired sentence of the row',
    )
    parser.add_argument(
        "--simcse_mlp_head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SimCSE-only: take the contrastive loss through Gao et al.'s "
        "Linear(d, d) + Tanh projection of the pooled vector, trained with the "
        "run and dropped at inference. --no-simcse_mlp_head takes it on the "
        "pooled vector itself",
    )
    parser.add_argument(
        "--student_pooling",
        choices=["cls", "mean"],
        default=None,
        help="GeoODE-KD/RKD/SimCSE: pooling of the student sentence vector "
        "(GeoODE-KD applies it at every layer)",
    )
    parser.add_argument(
        "--evaluate_test_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluate the test split, not the validation split, and skip validation "
            "entirely (default). No reported number is held out. "
            "--no-evaluate_test_each_epoch evaluates validation per epoch and keeps "
            "the final test score held out, which costs one extra evaluation pass"
        ),
    )
    parser.add_argument(
        "--pair_threshold_source",
        choices=["validation", "test"],
        default=None,
        help=(
            "Split used to sweep the pair-classification threshold before the final "
            'test evaluation. Follows --evaluate_test_each_epoch when unset. "test" '
            "(the default) calibrates on the test split itself, so its pair "
            "accuracy/F1 are an upper bound, not a held-out score"
        ),
    )
    parser.add_argument(
        "--no_eval_retrieval",
        action="store_true",
        help="Skip the ArguAna/FiQA/SCIDOCS nDCG@10 pass in the final test "
        "evaluation (it embeds ~92k documents)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Run the per-epoch evaluation every N epochs (0 disables it; only the final test "
        "evaluation runs, preceded by one validation pass when the pair thresholds "
        "are calibrated on validation)",
    )
    parser.add_argument(
        "--task_type",
        choices=["single_cls", "pair_cls", "pair_reg"],
        default=None,
        help="Training task contract",
    )

    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Optional durable directory for per-epoch student weights",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Shared directory for cached teacher embeddings (talas/geoode/rkd). The "
        "filename is derived from the teacher, pooling, max_length and the "
        "corpus contents, so runs of the same pair reuse one cache and runs of "
        "different pairs never collide. Overrides --cache_path",
    )
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Teacher embedding cache path"
    )

    parser.add_argument(
        "--fused_views",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Encode both dropout views in one forward over the doubled batch "
        "(default). --no-fused_views keeps the two-pass order, which is what runs "
        "recorded before this flag existed used",
    )
    parser.add_argument(
        "--cache_batch_size",
        type=int,
        default=None,
        help="Batch size of the one-off teacher caching pass (0 = use --batch_size)",
    )
    parser.add_argument(
        "--diag_every",
        type=int,
        default=None,
        help="Stride of the expensive training diagnostics: per-term gradient norms "
        "(weighted, so they say which term is actually driving the student), batch "
        "effective ranks, the signed H0 death-time residual and the student's own H1 "
        "diagram. 0 disables them; the cheap per-step diagnostics stay on either way. "
        "Nothing it computes is differentiated through, so a seeded run is unchanged",
    )
    parser.add_argument(
        "--probe_every",
        type=int,
        default=None,
        help="Stride of the structural probe: the audit ladder (Gram/CKA, k-NN "
        "overlap, mutual k-NN, H0 barcode, effective rank, TwoNN, anisotropy) on a "
        "fixed batch of corpus sentences, written to probe_metrics.jsonl. 0 is off. "
        "Needs a cached teacher (talas/geoode/rkd); encodes in eval() under no_grad, "
        "so a seeded run is unchanged",
    )
    parser.add_argument(
        "--probe_size",
        type=int,
        default=None,
        help="Sentences in the structural probe (seeded sample of training rows, "
        "whose teacher embeddings are already cached)",
    )
    parser.add_argument(
        "--probe_knn_k",
        type=int,
        default=None,
        help="Neighbourhood size of the probe's rung 3 (k-NN overlap and mutual k-NN)",
    )
    parser.add_argument(
        "--probe_seed", type=int, default=None, help="Seed of the probe sample"
    )
    parser.add_argument(
        "--weight_drift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log per-depth relative weight drift at the probe's cadence -- which "
        "layers the endpoint supervision actually reaches. Keeps an fp16 copy of the "
        "initial weights on the host",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )

    return parser.parse_args()


# CLI flags every method reads, and the config attribute each one sets. A flag
# left unset (None) never touches the config, so the method's own default stands.
COMMON_FLAGS = {
    "train_data": "train_data_path",
    "eval_data": "eval_data_path",
    "student_model": "student_model_name",
    "teacher_model": "teacher_model_name",
    "teacher_pooling": "pooling_method",
    "teacher_special_token": "teacher_special_token",
    "student_special_token": "student_special_token",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "save_every": "save_every",
    "lr": "learning_rate",
    "max_length": "max_length",
    "w_task": "w_task",
    "alpha_dtw": "alpha_dtw",
    "task_type": "task_type",
    "pair_threshold_source": "pair_threshold_source",
    "eval_every": "eval_every",
    "save_dir": "save_dir",
    "weights_dir": "weights_dir",
    "cache_path": "cache_path",
    "cache_dir": "cache_dir",
    "cache_batch_size": "cache_batch_size",
    "fused_views": "fused_views",
    "seed": "seed",
    "num_workers": "num_workers",
    "diag_every": "diag_every",
    "probe_every": "probe_every",
    "probe_size": "probe_size",
    "probe_knn_k": "probe_knn_k",
    "probe_seed": "probe_seed",
    "weight_drift": "log_weight_drift",
    "wandb_project": "wandb_project",
    "wandb_run_name": "wandb_run_name",
    "wandb_mode": "wandb_mode",
}

# Flags of a single method's own objective. Applied through apply_method_flags, so
# aiming one at a method that does not define it fails instead of being ignored.
METHOD_FLAGS = (
    (
        (
            "lambda_end",
            "lambda_ctr",
            "endpoint_loss",
            "lambda_gram",
            "lambda_topo",
            "lambda_h1",
            "topo_metric",
            "projection_type",
            "projection_seed",
            "learned_projector_lr_scale",
            "pca_center_fit",
            "pca_subtract_mean",
            "gauge_align",
            "gauge_rotation",
            "gauge_random_seed",
            "gauge_theta",
            "gauge_refit_every",
        ),
        "geoode method",
    ),
    (("w_dist", "w_angle", "normalize_student"), "rkd method"),
    (("simcse_view", "simcse_mlp_head"), "simcse method"),
    (("student_pooling",), "geoode, rkd and simcse methods"),
)

CONFIGS = {
    "cdm": CDMConfig,
    "dskd": DSKDConfig,
    "emo": EMOConfig,
    "stella": StellaConfig,
    "talas": TALASConfig,
    "geoode": GeoODEConfig,
    "rkd": RKDConfig,
    "simcse": SimCSEConfig,
}


def apply_method_flags(config, args, names, supported: str) -> None:
    """Copy the flags of one method's own objective onto its config.

    A flag whose attribute the selected config does not define is an error rather
    than a silently ignored argument: the run would otherwise print and log a
    setting it never applied.
    """
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            continue
        if not hasattr(config, name):
            raise ValueError(f"--{name} is only supported by the {supported}")
        setattr(config, name, value)


def get_config(method: str, args):
    config = CONFIGS.get(method, BaseConfig)()

    if args.save_every is not None and args.save_every <= 0:
        raise ValueError("--save_every must be a positive integer")
    if args.eval_every is not None and args.eval_every < 0:
        raise ValueError("--eval_every must be zero or positive")

    for flag, attribute in COMMON_FLAGS.items():
        value = getattr(args, flag)
        if value is not None:
            setattr(config, attribute, value)

    for names, supported in METHOD_FLAGS:
        apply_method_flags(config, args, names, supported)

    # The two eval flags describe one protocol, so each implies the other when only
    # one is given: a run either touches the validation split or it does not.
    if args.evaluate_test_each_epoch is not None:
        config.evaluate_test_each_epoch = args.evaluate_test_each_epoch
        if args.pair_threshold_source is None:
            config.pair_threshold_source = (
                "test" if args.evaluate_test_each_epoch else "validation"
            )
    elif args.pair_threshold_source == "validation":
        # Asking for a held-out threshold asks for the validation pass that selects it.
        config.evaluate_test_each_epoch = False
    if args.no_eval_retrieval:
        config.eval_retrieval = False
    if args.debug:
        config.debug_align = True
    if args.no_wandb:
        config.use_wandb = False

    return config


def main():
    args = parse_args()
    config = get_config(args.method, args)

    print("\n" + "=" * 70)
    print(f"Configuration for {args.method.upper()} method:")
    print("=" * 70)
    for key, value in config.to_dict().items():
        print(f"  {key:25s} : {value}")
    print("=" * 70 + "\n")

    # A failure here is fatal either way, so it is left to propagate: Python prints
    # the traceback and exits non-zero, which is what the hand-rolled handlers did.
    distiller = KnowledgeDistiller(config)
    try:
        distiller.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        distiller.close()


if __name__ == "__main__":
    main()
