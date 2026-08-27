import argparse
import sys

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
        '--method',
        type=str,
        default='cdm',
        choices=['cdm', 'dskd', 'emo', 'stella', 'talas', 'geoode', 'rkd', 'simcse'],
        help='Distillation method to use'
    )
    
    parser.add_argument(
        '--train_data',
        type=str,
        default=None,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        default=None,
        help='Path to evaluation data CSV file'
    )
    
    parser.add_argument(
        '--student_model',
        type=str,
        default=None,
        help='Student model name or path'
    )
    parser.add_argument(
        '--teacher_model',
        type=str,
        default=None,
        help='Teacher model name or path'
    )
    parser.add_argument(
        '--teacher_pooling',
        choices=['last_token', 'mean', 'cls'],
        default=None,
        help='Pooling of the teacher sentence vector: applied once at cache time by '
             'talas/geoode/rkd and every step by cdm/dskd/emo/stella. Qwen3-Embedding '
             'reads last_token, encoder teachers such as BGE-M3 read cls'
    )
    parser.add_argument(
        '--teacher_special_token',
        type=str,
        default=None,
        help='Sub-word marker of the teacher tokenizer that the CDM token alignment '
             'strips before comparing token strings ("Ġ" for Qwen3 byte-level BPE, '
             '"▁" for SentencePiece teachers such as BGE-M3). EMO reads it as the '
             'teacher BOS token string instead ("<s>")'
    )
    parser.add_argument(
        '--student_special_token',
        type=str,
        default=None,
        help='Sub-word marker of the student tokenizer ("##" for WordPiece students; '
             'EMO reads it as the student CLS token string, "[CLS]")'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=None,
        help='Save a periodic checkpoint every N epochs (must be positive)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--w_task',
        type=float,
        default=None,
        help='Task loss weight'
    )
    parser.add_argument(
        '--alpha_dtw',
        type=float,
        default=None,
        help='DTW KD loss weight'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='GeoODE-KD: weight of the instance-level semantic energy'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='GeoODE-KD: weight of the relational geometric energy'
    )
    parser.add_argument(
        '--lambda_end',
        type=float,
        default=None,
        help='GeoODE-KD: weight of the endpoint distillation loss'
    )
    parser.add_argument(
        '--lambda_dyn',
        type=float,
        default=None,
        help='GeoODE-KD: weight of the ODE consistency loss'
    )
    parser.add_argument(
        '--lambda_ctr',
        type=float,
        default=None,
        help='GeoODE-KD: weight of the contrastive regularizer'
    )
    parser.add_argument(
        '--pca_subtract_mean',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='GeoODE-KD: subtract the corpus mean before applying P_T (textbook PCA, '
             'removes the common component of the teacher embeddings)'
    )
    parser.add_argument(
        '--gauge_align',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='GeoODE-KD: Procrustes-align the PCA target coordinates to the untrained '
             'student (P_T = P_PCA R). --no-gauge_align is the ablation'
    )
    parser.add_argument(
        '--gauge_refit_every',
        type=int,
        default=None,
        help='GeoODE-KD: re-estimate the gauge R against the current student every N '
             'epochs (0 = keep the initial gauge)'
    )
    parser.add_argument(
        '--relational_target',
        choices=['native', 'projected'],
        default=None,
        help='GeoODE-KD: Gram matrix for the relational energy: the teacher\'s native '
             'cosine matrix, or that of the projected targets (ablation)'
    )
    parser.add_argument(
        '--guidance_schedule',
        choices=['linear', 'power', 'constant'],
        default=None,
        help='GeoODE-KD: depth-dependent guidance schedule s(t)'
    )
    parser.add_argument(
        '--guidance_power',
        type=float,
        default=None,
        help='GeoODE-KD: exponent p of the power guidance schedule s(t)=t^p'
    )
    parser.add_argument(
        '--w_dist',
        type=float,
        default=None,
        help='RKD: weight of the distance-wise relational loss (lambda_RKD-D)'
    )
    parser.add_argument(
        '--w_angle',
        type=float,
        default=None,
        help='RKD: weight of the angle-wise relational loss (lambda_RKD-A)'
    )
    parser.add_argument(
        '--normalize_student',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='RKD: L2-normalise the student embeddings before measuring relations, '
             'so they are compared on the same sphere as the normalised teacher '
             'cache. --no-normalize_student is the raw-Euclidean ablation'
    )
    parser.add_argument(
        '--simcse_view',
        choices=['dropout', 'pair'],
        default=None,
        help='SimCSE-only: positive view. "dropout" encodes the same sentence twice '
             '(unsupervised SimCSE); "pair" uses the paired sentence of the row'
    )
    parser.add_argument(
        '--student_pooling',
        choices=['cls', 'mean'],
        default=None,
        help='GeoODE-KD/RKD/SimCSE: pooling of the student sentence vector '
             '(GeoODE-KD applies it at every layer)'
    )
    parser.add_argument(
        '--evaluate_test_each_epoch',
        action='store_true',
        help=(
            'Evaluate the test split after every eval_every epochs instead of the '
            'validation split, and skip validation entirely. Requires '
            '--pair_threshold_source test; no reported number is held out'
        )
    )
    parser.add_argument(
        '--pair_threshold_source',
        choices=['validation', 'test'],
        default=None,
        help=(
            'Split used to sweep the pair-classification threshold before the final '
            'test evaluation. "test" calibrates on the test split itself, so its pair '
            'accuracy/F1 are an upper bound, not a held-out score'
        )
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=None,
        help='Run the per-epoch evaluation every N epochs (0 disables it; only the final test '
             'evaluation runs, preceded by one validation pass when the pair thresholds '
             'are calibrated on validation)'
    )
    parser.add_argument(
        '--depth_log_every',
        type=int,
        default=None,
        help='Sample per-depth diagnostics every N steps (0 disables; talas/geoode/rkd only)'
    )
    parser.add_argument(
        '--task_type',
        choices=['single_cls', 'pair_cls', 'pair_reg'],
        default=None,
        help='Training task contract'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--weights_dir',
        type=str,
        default=None,
        help='Optional durable directory for per-epoch student weights'
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default=None,
        help='Teacher embedding cache path'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name'
    )
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default=None,
        choices=['online', 'offline', 'disabled'],
        help='Weights & Biases mode'
    )
    
    return parser.parse_args()


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
    if method == 'cdm':
        config = CDMConfig()
    elif method == 'dskd':
        config = DSKDConfig()
    elif method == 'emo':
        config = EMOConfig()
    elif method == 'stella':
        config = StellaConfig()
    elif method == 'talas':
        config = TALASConfig()
    elif method == 'geoode':
        config = GeoODEConfig()
    elif method == 'rkd':
        config = RKDConfig()
    elif method == 'simcse':
        config = SimCSEConfig()
    else:
        config = BaseConfig()
    
    if args.train_data is not None:
        config.train_data_path = args.train_data
    if args.eval_data is not None:
        config.eval_data_path = args.eval_data
    
    if args.student_model is not None:
        config.student_model_name = args.student_model
    if args.teacher_model is not None:
        config.teacher_model_name = args.teacher_model
    if args.teacher_pooling is not None:
        config.pooling_method = args.teacher_pooling
    if args.teacher_special_token is not None:
        config.teacher_special_token = args.teacher_special_token
    if args.student_special_token is not None:
        config.student_special_token = args.student_special_token
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.save_every is not None:
        if args.save_every <= 0:
            raise ValueError("--save_every must be a positive integer")
        config.save_every = args.save_every
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_length is not None:
        config.max_length = args.max_length
    
    if args.w_task is not None:
        config.w_task = args.w_task
    if args.alpha_dtw is not None:
        config.alpha_dtw = args.alpha_dtw
    if args.task_type is not None:
        config.task_type = args.task_type
    if args.pair_threshold_source is not None:
        config.pair_threshold_source = args.pair_threshold_source
    if args.evaluate_test_each_epoch:
        config.evaluate_test_each_epoch = True
        # The flag's whole point is to run without a validation pass, so it carries
        # the threshold source with it unless one was named explicitly.
        if args.pair_threshold_source is None:
            config.pair_threshold_source = 'test'
    if args.eval_every is not None:
        if args.eval_every < 0:
            raise ValueError("--eval_every must be zero or positive")
        config.eval_every = args.eval_every
    if args.depth_log_every is not None:
        if args.depth_log_every < 0:
            raise ValueError("--depth_log_every must be zero or positive")
        config.depth_log_every = args.depth_log_every

    apply_method_flags(
        config,
        args,
        (
            'alpha',
            'beta',
            'lambda_end',
            'lambda_dyn',
            'lambda_ctr',
            'guidance_schedule',
            'guidance_power',
            'pca_subtract_mean',
            'gauge_align',
            'gauge_refit_every',
            'relational_target',
        ),
        'geoode method',
    )
    apply_method_flags(config, args, ('w_dist', 'w_angle', 'normalize_student'), 'rkd method')
    apply_method_flags(config, args, ('simcse_view',), 'simcse method')
    apply_method_flags(
        config, args, ('student_pooling',), 'geoode, rkd and simcse methods'
    )
    
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.weights_dir is not None:
        config.weights_dir = args.weights_dir
    if args.cache_path is not None:
        config.cache_path = args.cache_path
    
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug_align = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.wandb_mode is not None:
        config.wandb_mode = args.wandb_mode
    
    return config


def main():
    args = parse_args()
    
    config = get_config(args.method, args)
    
    print("\n" + "="*70)
    print(f"Configuration for {args.method.upper()} method:")
    print("="*70)
    for k, v in config.to_dict().items():
        print(f"  {k:25s} : {v}")
    print("="*70 + "\n")
    
    try:
        distiller = KnowledgeDistiller(config)
    except Exception as e:
        print(f"Error creating distiller: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        distiller.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        distiller.close()


if __name__ == '__main__':
    main()
