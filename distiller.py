import json
import os
import random
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler
from transformers import __version__ as transformers_version

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
try:
    from pytorch_optimizer import SAM

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print(
        "Warning: pytorch_optimizer not installed. SAM optimizer unavailable for TALAS."
    )
from src.cache_teacher import cache_teacher_embeddings, load_cached_embeddings
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.geoode_kd import GeoODEKD
from src.criterions.stella_distillation import (
    StellaModel,
    stella_stage1_loss,
    stella_stage2_loss,
)
from src.criterions.teacher_anchor_kd import TeacherAnchorKD
from src.data_utils import DualTokenizerCollate, TextPairRaw
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithTeacher,
    TextPairWithTeacher,
)

# Use evaluation_automodel for AutoModel (not evaluation_model_define which is for Stella)
from src.evaluation.evaluation_automodel import (
    eval_classification_task,
    eval_cls_tasks,
    eval_pair_task,
    eval_pair_tasks,
    eval_sts_task,
    eval_sts_tasks,
    test_cls_tasks,
    test_pair_tasks,
    test_sts_tasks,
)
from src.loss import info_nce
from src.pooling import last_token_pool, mean_pooling
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_pca_projection,
    project_teacher_embeddings,
)

# The distillation corpus (data/train_set/merged_3_data_5k_each.csv) is drawn from
# EMOTION, WiC and STS-B, so those three benchmarks are in-distribution and the
# remaining ones are held out. Reporting them as one number would let an
# in-distribution gain stand in for transfer, so the table averages them apart.
IOD_BENCHMARKS = frozenset({"emotion", "wic", "stsb"})


def should_save_epoch(epoch_index: int, save_every: int) -> bool:
    """Return whether a zero-based epoch is due for a periodic save."""
    return (epoch_index + 1) % save_every == 0


def is_finite(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and torch.isfinite(x).all().item()


def nonfinite_details(name: str, tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        return f"{name}: expected tensor, got {type(tensor).__name__}"
    if tensor.is_floating_point() or tensor.is_complex():
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
    else:
        nan_count = 0
        inf_count = 0
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, nan_count={nan_count}, inf_count={inf_count}"
    )


def assert_module_parameters_finite(module: nn.Module, module_name: str) -> None:
    finite_status = None
    for parameter in module.parameters():
        current = torch.isfinite(parameter).all()
        finite_status = current if finite_status is None else finite_status & current

    if finite_status is None or bool(finite_status.item()):
        return

    for name, parameter in module.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError(
                f"{module_name} parameters became NaN/Inf: "
                f"{nonfinite_details(name, parameter)}"
            )


def grads_are_finite(optim) -> bool:
    # Accumulated on device and read back once. Testing each gradient in a Python
    # `if` forces a host sync per parameter, which is ~200 stalls per step on a
    # BERT-base student -- for a check that is almost always True.
    finite_status = None
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            current = torch.isfinite(p.grad).all()
            finite_status = (
                current if finite_status is None else finite_status & current
            )
    return finite_status is None or bool(finite_status.item())


class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        self._validate_eval_config(config)
        self.wandb_run = None
        self.global_step = 0
        self.current_epoch = 0
        self.current_step = 0
        self._saved_checkpoint_epochs = set()
        # Per-depth diagnostics are sampled during the epoch and flushed once at the
        # end of it, like step_records: one file write per epoch, not one per step.
        self._depth_records: list[dict] = []
        self._depth_projection = None
        self.depth_probe = None
        self.depth_log_every = max(0, int(getattr(config, "depth_log_every", 0)))
        # Set before setup_training, which fills it in for the methods that need it.
        self.proj_s2t = None
        self.setup_seed(config.seed)
        self.setup_devices()
        self.setup_models()
        self.setup_data()
        self.setup_training()
        self.setup_wandb()

        # Initialize criterion based on method
        if config.distill_method == "cdm":
            self.criterion = ContextualDynamicMapping(
                tok_student=self.tok_student,
                tok_teacher=self.tok_teacher,
                blending_model_special_token=config.teacher_special_token,
                base_model_special_token=config.student_special_token,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw,
                debug_align=config.debug_align,
            )
        elif config.distill_method == "dskd":
            self.criterion = DualSpaceKD(
                student_dim=self.model_student.config.hidden_size,
                teacher_dim=self.model_teacher.config.hidden_size,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw,
            )
            # Move DSKD to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group(
                {"params": self.criterion.parameters(), "lr": config.learning_rate}
            )
            self.scheduler = self._build_scheduler()
            print("DSKD criterion initialized and added to optimizer")
        elif config.distill_method == "emo":
            self.criterion = EMODistillation(
                d_teacher=self.model_teacher.config.hidden_size,
                d_student=self.model_student.config.hidden_size,
                k_layers=getattr(config, "k_layers", 1),
                alpha_ot=getattr(config, "alpha_ot", 0.1),
                max_iter=getattr(config, "max_iter_ot", 100),
                teacher_special=getattr(config, "teacher_special_token", "<s>"),
                student_special=getattr(config, "student_special_token", "[CLS]"),
            )
            # Move EMO to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group(
                {"params": self.criterion.parameters(), "lr": config.learning_rate}
            )
            self.scheduler = self._build_scheduler()
            print("EMO criterion initialized and added to optimizer")
        elif config.distill_method == "geoode":
            # GeoODE-KD holds no parameters: the vector field is analytic in the
            # cached teacher targets, so nothing is added to the optimizer and the
            # deployed student is the unmodified encoder.
            self.criterion = GeoODEKD(
                alpha=config.alpha,
                beta=config.beta,
                lambda_end=config.lambda_end,
                lambda_dyn=config.lambda_dyn,
                lambda_ctr=config.lambda_ctr,
                contrastive_temperature=config.contrastive_temperature,
                guidance_schedule=config.guidance_schedule,
                guidance_power=config.guidance_power,
                pooling=config.student_pooling,
                include_embedding_layer=config.include_embedding_layer,
                stop_grad_target=config.stop_grad_target,
                eps_norm=config.eps_norm,
            ).to(self.device_s)
            self.depth_probe = self.criterion
            print(
                "GeoODE-KD criterion initialized: "
                f"alpha={config.alpha}, beta={config.beta}, "
                f"lambda_end={config.lambda_end}, lambda_dyn={config.lambda_dyn}, "
                f"lambda_ctr={config.lambda_ctr}, schedule={config.guidance_schedule}"
            )
        else:
            self.criterion = None

        # The depth diagnostics are parameter-free geometry, so any method with
        # per-layer hidden states and a batch teacher embedding can be measured with
        # the same probe. TALAS is measured too: hypothesis 1 of the paper is a
        # comparison against exactly that kind of static multi-layer anchoring, and
        # it cannot be checked from a GeoODE-only run.
        if self.depth_probe is None and self.depth_log_every > 0:
            if config.distill_method == "talas":
                self.depth_probe = GeoODEKD(
                    alpha=1.0,
                    beta=1.0,
                    pooling=getattr(config, "student_pooling", "cls"),
                    eps_norm=getattr(config, "eps_norm", 1e-12),
                ).to(self.device_s)
                print("Depth diagnostics probe enabled (alpha=1, beta=1)")
            else:
                # Online-teacher methods have no cached corpus to fit P_T on.
                print(
                    f"Depth diagnostics are not available for method "
                    f"{config.distill_method!r}; skipping"
                )

        # Metrics tracking
        self.step_times = []
        self.ma_window = deque(maxlen=50)
        self.warmup_steps = 10

    @staticmethod
    def _validate_eval_config(config) -> None:
        """Reject eval settings that can only fail once training is under way.

        Checked before the models load: a contradiction here otherwise surfaces as a
        caught exception at the end of epoch 1 and every epoch after it, leaving a run
        that trains for hours and reports nothing.
        """
        source = getattr(config, "pair_threshold_source", "validation")
        if source not in {"validation", "test"}:
            raise ValueError(
                f"Unsupported pair_threshold_source={source!r}; "
                "expected 'validation' or 'test'"
            )
        if getattr(config, "evaluate_test_each_epoch", False):
            if source != "test":
                raise ValueError(
                    "evaluate_test_each_epoch=True skips validation entirely, so the "
                    "pair threshold has nowhere to come from: set "
                    "pair_threshold_source='test' (CLI: --pair_threshold_source test)"
                )
            print(
                "Per-epoch evaluation runs on the TEST split and the pair threshold "
                "is swept on it. No validation pass will run, so no number this run "
                "reports is held out."
            )

    def setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Done setup_seed with seed={seed}")

    def setup_devices(self):
        if torch.cuda.device_count() >= 2:
            self.device_s = torch.device("cuda:0")  # student
            self.device_t = torch.device("cuda:1")  # teacher
            print(
                f"Using 2 GPUs: Student on {self.device_s}, Teacher on {self.device_t}"
            )
        elif torch.cuda.is_available():
            self.device_s = self.device_t = torch.device("cuda:0")
            print("[WARN] Only 1 GPU available -> both on cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_s = self.device_t = torch.device("mps")
            print("Using Apple Silicon MPS device")
        else:
            self.device_s = self.device_t = torch.device("cpu")
            print("[WARN] No GPU -> CPU training")
        print("Done setup_devices")

    def setup_wandb(self):
        cfg = self.config
        self.use_wandb = bool(getattr(cfg, "use_wandb", False))
        if not self.use_wandb:
            return
        if not WANDB_AVAILABLE:
            print(
                "Warning: wandb is not installed. Install requirements in the project venv to enable W&B logging."
            )
            self.use_wandb = False
            return
        mode = os.environ.get("WANDB_MODE", getattr(cfg, "wandb_mode", "online"))
        project = os.environ.get(
            "WANDB_PROJECT", getattr(cfg, "wandb_project", "iclr-mdd")
        )
        run_name = os.environ.get(
            "WANDB_RUN_NAME", getattr(cfg, "wandb_run_name", None)
        )
        self.wandb_run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            config=cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg),
        )
        print(f"W&B logging enabled: project={project}, run={run_name}, mode={mode}")

    @staticmethod
    def _flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
        flat = {}
        for key, value in values.items():
            name = f"{prefix}/{key}"
            if isinstance(value, (int, float)):
                flat[name] = float(value)
            elif isinstance(value, dict):
                flat.update(KnowledgeDistiller._flatten_metrics(name, value))
        return flat

    def setup_models(self):
        cfg = self.config

        print("Loading tokenizers...")
        tokenizer_kwargs = {"use_fast": True}
        self.tok_student = AutoTokenizer.from_pretrained(
            cfg.student_model_name,
            **tokenizer_kwargs,
        )
        self.tok_teacher = AutoTokenizer.from_pretrained(
            cfg.teacher_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if cfg.distill_method == "stella":
            print(f"Loading Stella student model: {cfg.student_model_name}")
            self.model_student = StellaModel(
                cfg.student_model_name,
                output_dim1=getattr(cfg, "output_dim1", 1024),
                pooling=getattr(cfg, "pooling", "cls"),
                output_dim2=getattr(cfg, "output_dim2", 512),
                output_dim3=getattr(cfg, "output_dim3", 256),
                output_dim4=getattr(cfg, "output_dim4", 128),
            )
            self.current_stage = 1
        else:
            print(f"Loading student model: {cfg.student_model_name}")
            student_kwargs = {}
            student_dtype_name = getattr(cfg, "student_dtype", None)
            student_dtypes = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if student_dtype_name is not None:
                if student_dtype_name not in student_dtypes:
                    raise ValueError(
                        f"Unsupported student_dtype={student_dtype_name!r}; "
                        f"expected one of {sorted(student_dtypes)}"
                    )
                try:
                    transformers_major = int(
                        transformers_version.split(".", maxsplit=1)[0]
                    )
                except (TypeError, ValueError):
                    transformers_major = 4
                dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
                student_kwargs[dtype_argument] = student_dtypes[student_dtype_name]

            # EMO reads the student's attention maps too, and SDPA returns none of
            # them: output_attentions=True then yields an empty list and the loss
            # indexes off the end of it.
            if cfg.distill_method == "emo":
                student_kwargs["attn_implementation"] = "eager"
                print(
                    "Using eager attention implementation for the EMO student "
                    "(required for output_attentions)"
                )
            self.model_student = AutoModel.from_pretrained(
                cfg.student_model_name,
                **student_kwargs,
            )

        print(f"Loading teacher model: {cfg.teacher_model_name}")
        teacher_kwargs = {"trust_remote_code": True}
        if cfg.teacher_dtype == "bfloat16":
            teacher_kwargs["torch_dtype"] = torch.bfloat16
        elif cfg.teacher_dtype == "float16":
            teacher_kwargs["torch_dtype"] = torch.float16

        # EMO method needs attentions, force eager attention implementation
        if cfg.distill_method == "emo":
            teacher_kwargs["attn_implementation"] = "eager"
            print(
                "Using eager attention implementation for the EMO teacher "
                "(required for output_attentions)"
            )

        self.model_teacher = AutoModel.from_pretrained(
            cfg.teacher_model_name, **teacher_kwargs
        )

        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)

        student_dtype = next(self.model_student.parameters()).dtype
        print(f"Student training dtype: {student_dtype}")
        assert_module_parameters_finite(self.model_student, "Student model after load")

        self.model_teacher.eval()
        for p in self.model_teacher.parameters():
            p.requires_grad_(False)

        print("Models loaded successfully!")
        print("Done setup_models")

    def setup_data(self):
        cfg = self.config

        print(f"Loading training data from: {cfg.train_data_path}")

        df = pd.read_csv(cfg.train_data_path)

        if cfg.task_type == "pair_cls":
            if "premise" not in df.columns or "hypothesis" not in df.columns:
                # Create from text column
                df["premise"] = df["text"] if "text" in df.columns else df.iloc[:, 0]
                df["hypothesis"] = df["text"] if "text" in df.columns else df.iloc[:, 0]

        self.task_head = None
        if cfg.distill_method == "emo":
            hidden_size = self.model_student.config.hidden_size
            if cfg.task_type == "single_cls" and "label" in df.columns:
                num_labels = int(df["label"].nunique())
                self.task_head = nn.Linear(hidden_size, num_labels).to(self.device_s)
            elif cfg.task_type == "pair_cls" and "label" in df.columns:
                num_labels = int(df["label"].nunique())
                self.task_head = nn.Linear(hidden_size * 4, num_labels).to(
                    self.device_s
                )

        # TALAS and GeoODE-KD both train against cached teacher embeddings only:
        # the teacher is run once, offline, and never during student optimization.
        if cfg.distill_method in ("talas", "geoode"):
            cache_path = Path(cfg.cache_path)

            # Check if cache exists
            if cache_path.exists():
                print(f"Loading cached teacher embeddings from: {cache_path}")
                teacher_cls_list = load_cached_embeddings(str(cache_path))
                print(f"Loaded {len(teacher_cls_list)} cached embeddings")
            else:
                print("Cache not found. Pre-computing teacher embeddings...")
                os.makedirs(cache_path.parent, exist_ok=True)

                # Create temporary dataset for caching
                temp_ds = TextPairRaw(df, cfg.task_type)
                temp_collate = DualTokenizerCollate(
                    self.tok_student, self.tok_teacher, cfg.task_type, cfg.max_length
                )
                cache_loader = DataLoader(
                    temp_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,  # Don't shuffle for caching
                    collate_fn=temp_collate,
                    pin_memory=True,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0,
                )

                # Cache teacher embeddings
                teacher_cls_list = cache_teacher_embeddings(
                    model_teacher=self.model_teacher,
                    dataloader=cache_loader,
                    device=self.device_t,
                    pooling_method=cfg.pooling_method,
                    normalize=cfg.normalize_cache,
                    dtype=torch.float32
                    if cfg.cache_dtype == "float32"
                    else torch.float16,
                    cache_path=str(cache_path),
                )
                print(
                    f"Cached {len(teacher_cls_list)} teacher embeddings to {cache_path}"
                )

            if len(teacher_cls_list) != len(df):
                raise ValueError(
                    f"Cached teacher embeddings length mismatch: cache has {len(teacher_cls_list)} "
                    f"rows but training data has {len(df)} rows. Remove or regenerate {cache_path}."
                )

            if cfg.distill_method == "geoode":
                teacher_cls_list = self._project_teacher_targets(
                    teacher_cls_list, df["premise"].astype(str).tolist()
                )
            elif self.depth_log_every > 0:
                # Diagnostics only: the training targets stay in the teacher space,
                # but the depth profile has to be measured in the student's. A
                # diagnostic must never take the training run down with it, so a
                # failure here disables the diagnostic instead of propagating.
                try:
                    self._depth_projection, _ = fit_pca_projection(
                        teacher_cls_list,
                        out_dim=self.model_student.config.hidden_size,
                        center=True,
                    )
                except Exception as error:  # noqa: BLE001
                    self.depth_log_every = 0
                    print(f"Depth diagnostics disabled: {error}")
                else:
                    print(
                        "Fitted a diagnostics-only PCA teacher projection "
                        f"{teacher_cls_list.shape[-1]} -> "
                        f"{self.model_student.config.hidden_size}"
                    )

            if cfg.distill_method == "geoode":
                # The gauge refit rewrites the targets in place between epochs; the
                # persistent DataLoader workers only see that through shared memory.
                teacher_cls_list = teacher_cls_list.contiguous().share_memory_()
            self.teacher_cls_all = teacher_cls_list

            # Free teacher model to save GPU memory (teacher not needed after caching)
            del self.model_teacher
            self.model_teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Teacher model freed from GPU memory")

            self.train_ds = TextPairWithTeacher(df, cfg.task_type, teacher_cls_list)
            self.collate_fn = DualTokenizerCollateWithTeacher(
                self.tok_student, cfg.task_type, cfg.max_length
            )
        else:
            # Standard distillation methods
            self.train_ds = TextPairRaw(df, cfg.task_type)

            self.collate_fn = DualTokenizerCollate(
                self.tok_student,
                self.tok_teacher,
                cfg.task_type,
                cfg.max_length,
            )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )

        print(f"Training samples: {len(self.train_ds)}")
        print(f"Training batches: {len(self.train_loader)}")
        print("Done setup_data")

    @torch.no_grad()
    def _student_initial_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Pooled, normalised final-layer embeddings of the *untrained* student."""
        cfg = self.config
        was_training = self.model_student.training
        self.model_student.eval()
        chunks = []
        for start in range(0, len(texts), 128):
            encoded = self.tok_student(
                texts[start : start + 128],
                max_length=cfg.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device_s)
            last = self.model_student(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            ).last_hidden_state
            if cfg.student_pooling == "mean":
                pooled = mean_pooling(last, encoded["attention_mask"])
            else:
                pooled = last[:, 0, :]
            chunks.append(F.normalize(pooled.float(), dim=-1).cpu())
        self.model_student.train(was_training)
        return torch.cat(chunks, dim=0)

    def _project_teacher_targets(
        self, teacher_cls: torch.Tensor, texts: list[str] | None = None
    ) -> torch.Tensor:
        """Fit and apply P_T = P_PCA R (Eq. 8), mapping cached teacher embeddings to d_S.

        Algorithm 1 does this once, before the training loop, so every step reads
        targets that already live on the student's hypersphere. P_PCA is fitted on
        the cached training embeddings; R is the orthogonal Procrustes alignment of
        those coordinates to the untrained student's embeddings of the same texts.
        Both are saved for reproducibility and neither is needed at inference.
        """
        cfg = self.config
        student_dim = self.model_student.config.hidden_size
        teacher_dim = teacher_cls.shape[-1]

        projection, mean = fit_pca_projection(
            teacher_cls,
            out_dim=student_dim,
            center=cfg.pca_center_fit,
        )
        targets = project_teacher_embeddings(
            teacher_cls,
            projection,
            mean=mean,
            subtract_mean=cfg.pca_subtract_mean,
            eps=cfg.eps_norm,
        )

        explained = 1.0
        if teacher_dim <= student_dim:
            print(
                f"Teacher dim {teacher_dim} <= student dim {student_dim}: "
                "P_T is the identity, targets are only re-normalized"
            )
        else:
            # Eckart-Young: among all rank-d_S linear maps, PCA retains the largest
            # share of the cached embedding energy, i.e. it is the linear map that
            # best preserves the Gram matrix E_geo is defined over. This number is
            # what the paper reports for P_T.
            cached = teacher_cls.to(torch.float32)
            explained = float(
                (cached @ projection).pow(2).sum() / cached.pow(2).sum().clamp(min=1e-12)
            )
            print(
                f"Fitted PCA teacher projection {teacher_dim} -> {student_dim} "
                f"(retains {explained:.1%} of cached embedding energy)"
            )

        rotation = None
        gauge_stats = None
        if getattr(cfg, "gauge_align", False):
            if texts is None:
                raise ValueError("gauge_align requires the corpus texts")
            n_fit = min(len(texts), int(getattr(cfg, "gauge_align_samples", 16384)))
            if n_fit < 2 * student_dim:
                print(
                    f"Gauge alignment skipped: {n_fit} sentences is too few for a "
                    f"{student_dim}-dimensional Procrustes fit"
                )
            else:
                # Evenly spaced rows so every source in a merged corpus is represented.
                index = torch.linspace(0, len(texts) - 1, n_fit).round().long().unique()
                student_init = self._student_initial_embeddings(
                    [texts[i] for i in index.tolist()]
                )
                rotation, gauge_stats = fit_gauge_alignment(targets[index], student_init)
                # Kept for the optional per-epoch re-estimation of R (alternating
                # minimisation of the gauge-invariant endpoint discrepancy).
                self._gauge_state = {
                    "targets_pca": targets.clone(),
                    "index": index,
                    "texts": [texts[i] for i in index.tolist()],
                    "history": [gauge_stats],
                }
                targets = F.normalize(targets @ rotation, dim=-1, eps=cfg.eps_norm)
                print(
                    "Fitted Procrustes gauge alignment R on "
                    f"{gauge_stats['samples']} sentences: mean student-target cosine "
                    f"{gauge_stats['cos_before']:+.3f} -> {gauge_stats['cos_after']:+.3f}"
                )

        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            projection_path = os.path.join(cfg.save_dir, "teacher_projection.pt")
            torch.save(
                {
                    "projection": projection,
                    "mean": mean,
                    "teacher_model_name": cfg.teacher_model_name,
                    "student_dim": student_dim,
                    "teacher_dim": teacher_dim,
                    "pca_center_fit": cfg.pca_center_fit,
                    "pca_subtract_mean": cfg.pca_subtract_mean,
                    "explained_energy": explained,
                    "gauge_rotation": rotation,
                    "gauge_stats": gauge_stats,
                },
                projection_path,
            )
            print(f"Teacher projection saved: {projection_path}")

        return targets

    @torch.no_grad()
    def _refit_gauge(self, epoch: int) -> None:
        """Re-estimate R against the *current* student (closed-form Procrustes on the
        same corpus subset) and rewrite the targets in place.

        With the student fixed this is the exact minimiser over O(d_S) of the
        endpoint discrepancy, and with R fixed the optimiser lowers it in theta, so
        the alternation descends min_{theta, R} E_sem(Z_theta, T R) monotonically.
        R is orthogonal, so the targets' Gram matrix is untouched: only the gauge
        moves, never the geometry.
        """
        state = getattr(self, "_gauge_state", None)
        if state is None:
            return
        student_now = self._student_initial_embeddings(state["texts"])
        subset = state["targets_pca"][state["index"]]
        rotation, stats = fit_gauge_alignment(subset, student_now)
        # Cosine the student currently has against the targets it was trained on.
        stats["cos_previous_gauge"] = float(
            (self.teacher_cls_all[state["index"]] * student_now).sum(dim=-1).mean()
        )
        stats["epoch"] = epoch + 1
        new_targets = F.normalize(
            state["targets_pca"] @ rotation, dim=-1, eps=self.config.eps_norm
        )
        self.teacher_cls_all.copy_(new_targets.to(self.teacher_cls_all.dtype))
        state["history"].append(stats)
        print(
            f"Gauge refit after epoch {epoch + 1}: student-target cosine under the "
            f"previous R {stats['cos_previous_gauge']:+.4f} -> under the refit R "
            f"{stats['cos_after']:+.4f}"
        )
        if self.config.save_dir:
            path = os.path.join(self.config.save_dir, "teacher_projection.pt")
            if os.path.exists(path):
                saved = torch.load(path, map_location="cpu")
                saved["gauge_rotation"] = rotation
                saved["gauge_history"] = state["history"]
                torch.save(saved, path)

    def _should_log_depth(self, step: int) -> bool:
        if self.depth_log_every <= 0 or self.depth_probe is None:
            return False
        return step == 0 or (step + 1) % self.depth_log_every == 0

    def _depth_teacher_targets(self, teacher_cls: torch.Tensor) -> torch.Tensor:
        """Teacher targets on the student's hypersphere, for diagnostics."""
        targets = teacher_cls.float()
        projection = self._depth_projection
        if projection is not None:
            if targets.shape[-1] != projection.shape[0]:
                raise ValueError(
                    f"depth projection expects {projection.shape[0]}-dimensional "
                    f"teacher embeddings but the batch carries {targets.shape[-1]}; "
                    "the projection is fitted only for methods whose training targets "
                    "stay in the teacher space"
                )
            targets = targets @ projection.to(targets.device, targets.dtype)
        return F.normalize(targets, p=2, dim=-1, eps=1e-12)

    @torch.no_grad()
    def _record_depth(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None,
        teacher_cls: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Sample one per-depth report and buffer it for the end of the epoch."""
        states = self.depth_probe.layer_states(hidden_states, attention_mask)
        report = self.depth_probe.depth_report(
            states, self._depth_teacher_targets(teacher_cls)
        )
        self._depth_records.append(
            {
                "method": self.config.distill_method,
                "epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "step": self.current_step,
                "batch_size": int(batch_size),
                **report,
            }
        )

    def _build_scheduler(self):
        cfg = self.config
        total_steps = len(self.train_loader) * cfg.epochs
        min_lr_rate = cfg.min_lr / cfg.learning_rate
        return get_scheduler(
            name="cosine_with_min_lr",
            optimizer=self.optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
            scheduler_specific_kwargs={"min_lr_rate": min_lr_rate},
        )

    def setup_training(self):
        cfg = self.config

        # TALAS optimizer/scheduler will be initialized after criterion creation in train_step
        if cfg.distill_method == "talas":
            self.optimizer = None
            self.scheduler = None
            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
            print(
                "TALAS: Deferring optimizer/scheduler initialization until criterion is created"
            )
        else:
            optimizer_parameters = list(self.model_student.parameters())
            if self.task_head is not None:
                optimizer_parameters.extend(self.task_head.parameters())
            param_groups = [
                {"params": optimizer_parameters, "lr": cfg.learning_rate}
            ]

            # CDM maps the student CLS into the teacher space. The projection is
            # built here rather than on the first step because a param group added
            # after the scheduler was constructed has no matching base_lr, and
            # scheduler.step() then fails on the length mismatch.
            if cfg.distill_method == "cdm":
                d_s = self.model_student.config.hidden_size
                d_t = self.model_teacher.config.hidden_size
                self.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(self.device_s)
                param_groups.append(
                    {
                        "params": self.proj_s2t.parameters(),
                        "lr": cfg.learning_rate * 2,
                    }
                )
                print(f"Initialized projection layer: {d_s} -> {d_t}")

            self.optimizer = optim.AdamW(param_groups, lr=cfg.learning_rate)

            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

            self.scheduler = self._build_scheduler()

        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            print(f"Checkpoints will be saved to: {cfg.save_dir}")
        print("Done setup_training")

    def sync_all(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

    def _compute_task_loss(
        self,
        student_cls1: torch.Tensor,
        student_cls2: torch.Tensor | None,
        batch_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        labels = batch_s.get("labels")
        if cfg.task_type == "single_cls":
            if labels is None or self.task_head is None:
                raise ValueError("single_cls training requires labels and a task head")
            logits = self.task_head(student_cls1)
            loss = F.cross_entropy(logits, labels.long())
            return loss, {
                "task_accuracy": float(
                    (logits.argmax(-1) == labels).float().mean().item()
                )
            }

        if student_cls2 is None:
            raise ValueError(f"{cfg.task_type} training requires a second text")

        if (
            cfg.task_type == "pair_cls"
            and labels is not None
            and self.task_head is not None
        ):
            pair_features = torch.cat(
                [
                    student_cls1,
                    student_cls2,
                    torch.abs(student_cls1 - student_cls2),
                    student_cls1 * student_cls2,
                ],
                dim=-1,
            )
            logits = self.task_head(pair_features)
            loss = F.cross_entropy(logits, labels.long())
            return loss, {
                "task_accuracy": float(
                    (logits.argmax(-1) == labels).float().mean().item()
                )
            }

        if cfg.task_type == "pair_reg" and labels is not None:
            cosine = F.cosine_similarity(student_cls1, student_cls2)
            predictions = (cosine + 1.0) * 2.5
            loss = F.mse_loss(predictions, labels.float())
            return loss, {"task_mse": float(loss.detach().item())}

        loss, _ = info_nce(student_cls1, student_cls2, temperature=cfg.temperature)
        return loss, {}

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        method = cfg.distill_method

        if method == "talas":
            batch_s = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_stu") or k == "labels" or k == "teacher_cls":
                    batch_s[k] = v.to(self.device_s, non_blocking=True)

            # ========== FIRST PASS ==========
            with autocast("cuda", enabled=torch.cuda.is_available()):
                teacher_cls = batch_s["teacher_cls"]

                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True,
                )

                S_last1 = s_out1.last_hidden_state
                S_last2 = s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = S_last2[:, 0, :]

                loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)

                # Initialize TALAS criterion if needed
                if self.criterion is None:
                    d_s = self.model_student.config.hidden_size
                    d_t = teacher_cls.shape[-1]

                    # BERT-base has 13 layers: embedding + 12 transformer layers
                    num_layers = len(s_out1.hidden_states)

                    self.criterion = TeacherAnchorKD(
                        student_dim=d_s,
                        teacher_dim=d_t,
                        num_layers=num_layers,
                        last_layer_idx=cfg.last_layer_idx,
                        start_rkd=cfg.start_rkd,
                        w_task=cfg.w_task,
                        w_kd=cfg.w_kd,
                        w_struct=cfg.w_struct,
                        eps_norm=cfg.eps_norm,
                    ).to(self.device_s)

                    # Initialize SAM optimizer with both student and criterion parameters
                    if not SAM_AVAILABLE:
                        raise RuntimeError(
                            "SAM optimizer not available. Install pytorch_optimizer."
                        )

                    base_optimizer = optim.AdamW
                    self.optimizer = SAM(
                        [
                            {
                                "params": self.model_student.parameters(),
                                "lr": cfg.learning_rate,
                                "weight_decay": 0.01,
                            },
                            {
                                "params": self.criterion.parameters(),
                                "lr": cfg.learning_rate * 5,
                            },
                        ],
                        base_optimizer,
                        rho=getattr(cfg, "rho", 0.05),
                        adaptive=True,
                    )

                    # Initialize scheduler
                    num_steps = len(self.train_loader)
                    total_steps = num_steps * cfg.epochs
                    min_lr_rate = cfg.min_lr / cfg.learning_rate
                    self.scheduler = get_scheduler(
                        name="cosine_with_min_lr",
                        optimizer=self.optimizer,
                        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                        num_training_steps=total_steps,
                        scheduler_specific_kwargs={"min_lr_rate": min_lr_rate},
                    )

                    print(
                        f"Initialized TeacherAnchorKD: {d_s} -> {d_t}, num_layers={num_layers}, last_layer_idx={cfg.last_layer_idx}, start_rkd={cfg.start_rkd}"
                    )
                    print(
                        f"Initialized SAM optimizer with rho={getattr(cfg, 'rho', 0.05)}"
                    )
                    print(
                        f"Initialized scheduler: {total_steps} steps, warmup={int(total_steps * cfg.warmup_ratio)}"
                    )

                # Now safe to call criterion with initialized projection heads
                student_outputs = {
                    "hidden_states": s_out1.hidden_states,
                    "last_hidden_state": S_last1,
                }

                loss, metrics = self.criterion(
                    student_outputs=student_outputs,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task,
                )

                loss = loss.float()

            # Backward pass 1 (this will init gradients for first_step)
            self.scaler.scale(loss).backward()

            # Check gradients
            self.scaler.unscale_(self.optimizer)
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, "skip": "grad_inf_p1"}

            # SAM first step
            self.optimizer.first_step(zero_grad=True)

            # ========== SECOND PASS ==========
            with autocast("cuda", enabled=torch.cuda.is_available()):
                s_out1_2 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                s_out2_2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True,
                )

                S_last1_2 = s_out1_2.last_hidden_state
                S_last2_2 = s_out2_2.last_hidden_state
                S_cls1_2 = S_last1_2[:, 0, :]
                S_cls2_2 = S_last2_2[:, 0, :]

                loss_task_2, _ = info_nce(
                    S_cls1_2, S_cls2_2, temperature=cfg.temperature
                )

                student_outputs_2 = {
                    "hidden_states": s_out1_2.hidden_states,
                    "last_hidden_state": S_last1_2,
                }

                loss_2, _ = self.criterion(
                    student_outputs=student_outputs_2,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task_2,
                )

                loss_2 = loss_2.float()

            # Check loss_2 is finite
            if not is_finite(loss_2):
                raise RuntimeError("loss_2 NaN/Inf")

            # Check loss_2 finite before backward
            if not is_finite(loss_2):
                raise RuntimeError(
                    f"loss_2 NaN/Inf at epoch={self.current_epoch} step={self.current_step}"
                )

            # Backward pass 2 - IMPORTANT: Do NOT scale (plain backward)
            loss_2.backward()

            # Check gradients again
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, "skip": "grad_inf_p2"}

            # SAM second step
            self.optimizer.second_step(zero_grad=True)
            self.scaler.update()
            self.scheduler.step()

            if self._should_log_depth(self.current_step):
                self._record_depth(
                    s_out1_2.hidden_states,
                    batch_s["attention_mask1_stu"],
                    teacher_cls,
                    batch_s["input_ids1_stu"].size(0),
                )

            # Clean up
            del s_out1, s_out2, s_out1_2, s_out2_2
            del student_outputs, student_outputs_2

            return loss, metrics

        if method == "geoode":
            batch_s = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_stu") or k in ("labels", "teacher_cls"):
                    batch_s[k] = v.to(self.device_s, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                teacher_cls = batch_s["teacher_cls"]
                input_ids = batch_s["input_ids1_stu"]
                attention_mask = batch_s["attention_mask1_stu"]

                s_out = self.model_student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Eq. (37) needs a second view of the *same* sentence. The default
                # runs the encoder twice so the two views differ only by dropout;
                # "pair" instead reuses the paired sentence already in the batch.
                second_view = None
                if cfg.lambda_ctr > 0:
                    if cfg.contrastive_view == "dropout":
                        view_ids, view_mask = input_ids, attention_mask
                    elif cfg.contrastive_view == "pair":
                        view_ids = batch_s["input_ids2_stu"]
                        view_mask = batch_s["attention_mask2_stu"]
                    else:
                        raise ValueError(
                            f"Unsupported contrastive_view={cfg.contrastive_view!r}; "
                            "expected 'dropout' or 'pair'"
                        )
                    s_out_view = self.model_student(
                        input_ids=view_ids,
                        attention_mask=view_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    view_last = s_out_view.last_hidden_state
                    if cfg.student_pooling == "mean":
                        second_view = mean_pooling(view_last, view_mask)
                    else:
                        second_view = view_last[:, 0, :]

                loss, metrics = self.criterion(
                    hidden_states=s_out.hidden_states,
                    teacher=teacher_cls,
                    attention_mask=attention_mask,
                    second_view=second_view,
                )
                loss = loss.float()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self._should_log_depth(self.current_step):
                self._record_depth(
                    s_out.hidden_states,
                    attention_mask,
                    teacher_cls,
                    input_ids.size(0),
                )

            return loss, metrics

        # Standard distillation methods with teacher inference
        batch_s, batch_t = {}, {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                continue
            if k.endswith("_stu") or k == "labels":
                batch_s[k] = v.to(self.device_s, non_blocking=True)
            if k.endswith("_tea"):
                batch_t[k] = v.to(self.device_t, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            need_atts = method == "emo"
            # no_grad, not inference_mode: these teacher tensors are consumed by
            # losses that save them for backward, and inference tensors cannot be.
            with torch.no_grad():
                t_out1 = self.model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    output_attentions=need_atts,
                    return_dict=True,
                )
                T_last1 = t_out1.last_hidden_state
                T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])

                T_last1 = T_last1.to(self.device_s, non_blocking=True)
                T_cls1 = T_cls1.to(self.device_s, non_blocking=True)

                if need_atts:
                    T_atts = tuple(
                        att.to(self.device_s, non_blocking=True)
                        for att in t_out1.attentions
                    )
                    T_last2 = None
                    T_atts2 = None
                    if "input_ids2_tea" in batch_t:
                        t_out2 = self.model_teacher(
                            input_ids=batch_t["input_ids2_tea"],
                            attention_mask=batch_t["attention_mask2_tea"],
                            output_attentions=True,
                            return_dict=True,
                        )
                        T_last2 = t_out2.last_hidden_state.to(
                            self.device_s, non_blocking=True
                        )
                        T_atts2 = tuple(
                            attention.to(self.device_s, non_blocking=True)
                            for attention in t_out2.attentions
                        )

            # Different models have different forward signatures
            if method == "stella":
                # StellaModel doesn't accept output_attentions or return_dict
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                )
            elif method == "emo":
                # EMO needs attentions
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_attentions=True,
                    return_dict=True,
                )
                s_out2 = None
                if "input_ids2_stu" in batch_s:
                    s_out2 = self.model_student(
                        input_ids=batch_s["input_ids2_stu"],
                        attention_mask=batch_s["attention_mask2_stu"],
                        output_attentions=True,
                        return_dict=True,
                    )
            else:
                # CDM, DSKD - standard transformers models
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    return_dict=True,
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    return_dict=True,
                )
            if method != "stella":
                S_last1 = s_out1.last_hidden_state
                S_last2 = None if s_out2 is None else s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = None if S_last2 is None else S_last2[:, 0, :]
            else:
                S_cls1 = s_out1["pooled"]
                S_cls2 = s_out2["pooled"]

            if method == "emo":
                loss_task, task_metrics = self._compute_task_loss(
                    S_cls1, S_cls2, batch_s
                )
            else:
                loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
                task_metrics = {}

            # ========== Method-specific KD loss ==========
            if method == "cdm":
                keep_s1 = batch_s["attention_mask1_stu"].bool() & (
                    ~batch_s["special_tokens_mask1_stu"].bool()
                )
                keep_t1 = batch_t["attention_mask1_tea"].to(self.device_s).bool() & (
                    ~batch_t["special_tokens_mask1_tea"].to(self.device_s).bool()
                )

                kd_dtw = self.criterion.compute_cdm_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    batch_input_ids_stu=batch["input_ids1_stu"],
                    batch_input_ids_tea=batch["input_ids1_tea"],
                    keep_mask_stu=keep_s1,
                    keep_mask_tea=keep_t1,
                    proj_s2t=self.proj_s2t,
                    device_s=self.device_s,
                    epoch=self.current_epoch,
                    step=self.current_step,
                )

                S_proj_cls1 = self.proj_s2t(S_cls1)
                S_proj_cls1_norm = F.normalize(S_proj_cls1, p=2, dim=-1)
                T_cls1_norm = F.normalize(T_cls1, p=2, dim=-1)
                kd_cls = F.mse_loss(S_proj_cls1_norm, T_cls1_norm)

                loss = (
                    cfg.w_task * loss_task
                    + cfg.alpha_dtw * kd_dtw * 100
                    + cfg.w_cls * kd_cls
                )

                metrics = {
                    "loss_total": loss.item(),
                    "loss_task": loss_task.item(),
                    "loss_kd_dtw": kd_dtw.item()
                    if isinstance(kd_dtw, torch.Tensor)
                    else kd_dtw,
                    "loss_kd_cls": kd_cls.item(),
                }

            elif method == "dskd":
                mask_s1 = batch_s["attention_mask1_stu"]
                mask_t1 = batch_t["attention_mask1_tea"].to(self.device_s)

                spec_s1 = batch_s.get("special_tokens_mask1_stu", None)
                spec_t1 = batch_t.get("special_tokens_mask1_tea", None)
                if spec_t1 is not None:
                    spec_t1 = spec_t1.to(self.device_s)

                loss, metrics = self.criterion.compute_dskd_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    S_cls=S_cls1,
                    T_cls=T_cls1,
                    mask_student=mask_s1,
                    mask_teacher=mask_t1,
                    task_loss=loss_task,
                    special_tokens_mask_student=spec_s1,
                    special_tokens_mask_teacher=spec_t1,
                    device=self.device_s,
                )

            elif method == "emo":

                class TeacherOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions

                class StudentOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions

                teacher_outputs = TeacherOutput(T_last1, T_atts)
                student_outputs = StudentOutput(S_last1, s_out1.attentions)

                att_loss_weight = getattr(cfg, "att_loss_weight", 0.1)
                ot_loss_weight = getattr(cfg, "ot_loss_weight", 1.0)

                kd_loss, kd_metrics = self.criterion.compute_emo_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                    input_ids_tea=batch_t["input_ids1_tea"].to(self.device_s),
                    input_ids_stu=batch_s["input_ids1_stu"],
                    attention_mask_tea=batch_t["attention_mask1_tea"].to(self.device_s),
                    attention_mask_stu=batch_s["attention_mask1_stu"],
                    tok_teacher=self.tok_teacher,
                    tok_student=self.tok_student,
                    att_loss_weight=att_loss_weight,
                    ot_loss_weight=ot_loss_weight,
                )
                if S_last2 is not None and T_last2 is not None:
                    teacher_outputs2 = TeacherOutput(T_last2, T_atts2)
                    student_outputs2 = StudentOutput(S_last2, s_out2.attentions)
                    kd_loss2, kd_metrics2 = self.criterion.compute_emo_loss(
                        teacher_outputs=teacher_outputs2,
                        student_outputs=student_outputs2,
                        input_ids_tea=batch_t["input_ids2_tea"].to(self.device_s),
                        input_ids_stu=batch_s["input_ids2_stu"],
                        attention_mask_tea=batch_t["attention_mask2_tea"].to(
                            self.device_s
                        ),
                        attention_mask_stu=batch_s["attention_mask2_stu"],
                        tok_teacher=self.tok_teacher,
                        tok_student=self.tok_student,
                        att_loss_weight=att_loss_weight,
                        ot_loss_weight=ot_loss_weight,
                    )
                    kd_loss = 0.5 * (kd_loss + kd_loss2)
                    kd_metrics = {
                        key: 0.5 * (kd_metrics[key] + kd_metrics2[key])
                        for key in kd_metrics
                    }

                w_task = getattr(cfg, "w_task", 0.5)
                alpha_kd = getattr(cfg, "alpha_kd", 0.5)
                loss = w_task * loss_task + alpha_kd * kd_loss

                metrics = {
                    "loss_total": loss.item(),
                    "loss_task": loss_task.item(),
                    **task_metrics,
                    **kd_metrics,
                }

            elif method == "stella":
                if self.current_stage == 1:
                    S_emb = s_out1["fc1"]
                    loss, metrics = stella_stage1_loss(
                        S_emb,
                        T_cls1,
                        w_cos=getattr(cfg, "w_cos_stage1", 10.0),
                        w_sim=getattr(cfg, "w_sim_stage1", 200.0),
                        w_tri=getattr(cfg, "w_tri_stage1", 20.0),
                    )
                else:
                    loss, metrics = stella_stage2_loss(
                        S_cls1,
                        S_cls2,
                        s_out1["fc1"],
                        s_out1["fc2"],
                        s_out1["fc3"],
                        s_out1["fc4"],
                        T_cls1,
                        temperature=cfg.temperature,
                        w_task=cfg.w_task,
                        w_cos=getattr(cfg, "w_cos_stage2", 10.0),
                        w_sim=getattr(cfg, "w_sim_stage2", 200.0),
                        w_tri=getattr(cfg, "w_tri_stage2", 20.0),
                    )

            else:
                raise ValueError(f"Unknown distillation method: {method}")

            loss = loss.float()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss, metrics

    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch

        total_loss = 0.0
        n_items = 0
        metric_totals = {}
        epoch_step_times = []
        peak_memory_mb = 0.0
        # Per-step diagnostics, buffered here and written once at the end of the epoch.
        # Epoch means alone cannot show *when* inside an epoch a curve flattened, so
        # five points per curve is a summary, not a diagnosis. Buffering keeps this to
        # one file write per epoch rather than one per step.
        step_records: list[dict] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
        )

        for step, batch in enumerate(pbar):
            self.current_step = step

            self.sync_all()
            t0 = time.perf_counter()

            loss, metrics = self.train_step(batch)

            self.sync_all()
            dt = time.perf_counter() - t0
            epoch_step_times.append(dt)
            self.global_step += 1
            if getattr(self, "use_wandb", False) and WANDB_AVAILABLE:
                log_payload = {
                    "train/epoch": epoch + 1,
                    "train/global_step": self.global_step,
                    "train/step_seconds": dt,
                }
                log_payload.update(self._flatten_metrics("train", metrics))
                wandb.log(log_payload, step=self.global_step)

            bs = batch["input_ids1_stu"].size(0)
            loss_value = loss.item()
            total_loss += loss_value * bs
            n_items += bs
            avg_loss = total_loss / max(1, n_items)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * bs

            step_record = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "step": step,
                "batch_size": int(bs),
                "loss": float(loss_value),
                "step_seconds": float(dt),
                # train_step() has already called scheduler.step(), so this is the rate
                # the *next* step will use.
                "lr_next": float(self.optimizer.param_groups[0]["lr"]),
            }
            step_record.update(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            step_records.append(step_record)

            mem_info = {}
            for dev_id in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
                peak_memory_mb = max(peak_memory_mb, mem_alloc)
                mem_info[f"gpu{dev_id}"] = f"{mem_alloc:.0f}/{mem_reserved:.0f}MB"

            if step >= self.warmup_steps:
                self.step_times.append(dt)
                self.ma_window.append(dt)
                avg_step = sum(self.step_times) / len(self.step_times)
                ma_step = sum(self.ma_window) / len(self.ma_window)

                postfix = {
                    "avg_loss": f"{avg_loss:.4f}",
                    "ms/step": f"{avg_step * 1000:.1f}",
                    "ms/step(ma)": f"{ma_step * 1000:.1f}",
                    "it/s": f"{1.0 / ma_step:.2f}",
                    **mem_info,
                }

                for k, v in metrics.items():
                    if k != "loss_total":
                        # Format only if v is numeric (not string like 'skip': 'grad_inf_p1')
                        postfix[k] = (
                            f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                        )

                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", **mem_info})

        avg_loss = total_loss / max(1, n_items)

        if len(self.step_times) > 0:
            epoch_avg = sum(self.step_times) / len(self.step_times)
            print(
                f"[Epoch {epoch + 1}] Avg step time = {epoch_avg * 1000:.2f} ms "
                f"({1.0 / epoch_avg:.2f} it/s)"
            )

        print(f"Done train_epoch {epoch + 1}")
        self.log_step_records(step_records)
        depth_summary = self.log_depth_records()
        epoch_means = {
            key: value / max(1, n_items) for key, value in metric_totals.items()
        }

        # The progress bar shows the *last* step's diagnostics, and the last batch is
        # usually a short remainder, so its numbers swing for reasons that have nothing
        # to do with training. Print the example-weighted epoch means instead.
        if epoch_means:
            shown = ["loss_total"] if "loss_total" in epoch_means else []
            shown += sorted(k for k in epoch_means if k not in shown)
            body = "  ".join(f"{k}={epoch_means[k]:.4f}" for k in shown)
            print(f"[Epoch {epoch + 1}] mean over {n_items} examples: {body}")

        self.last_epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "mean_step_seconds": (
                sum(epoch_step_times) / len(epoch_step_times)
                if epoch_step_times
                else 0.0
            ),
            "peak_memory_mb": peak_memory_mb,
            **epoch_means,
        }
        if depth_summary is not None:
            self.last_epoch_metrics["depth"] = depth_summary
            if getattr(self, "use_wandb", False) and WANDB_AVAILABLE:
                payload = self._flatten_metrics("depth", depth_summary)
                payload.update(
                    {
                        f"depth/{key}_l{index + 1}": value
                        for key in ("cos_teacher", "gram_gap", "energy")
                        for index, value in enumerate(depth_summary[key])
                    }
                )
                wandb.log(payload, step=self.global_step)
        return avg_loss

    def save_checkpoint(self, epoch: int, metrics: dict | None = None):
        cfg = self.config
        if not cfg.save_dir:
            return
        if epoch in self._saved_checkpoint_epochs:
            print(
                f"Checkpoint for epoch {epoch + 1} already saved; skipping duplicate."
            )
            return
        os.makedirs(cfg.save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        }

        if self.proj_s2t is not None:
            checkpoint["proj_s2t_state_dict"] = self.proj_s2t.state_dict()

        if self.criterion is not None and hasattr(self.criterion, "state_dict"):
            checkpoint["criterion_state_dict"] = self.criterion.state_dict()
        if self.task_head is not None:
            checkpoint["task_head_state_dict"] = self.task_head.state_dict()

        if metrics:
            checkpoint["metrics"] = metrics

        path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        print(f"Done save_checkpoint for epoch {epoch + 1}")

        if cfg.save_best and metrics and "loss" in metrics:
            if not hasattr(self, "best_loss") or metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")

        self.save_student_weights(epoch)
        self._saved_checkpoint_epochs.add(epoch)

    def save_student_weights(self, epoch: int):
        weights_dir = getattr(self.config, "weights_dir", None)
        if not weights_dir:
            return

        destination_dir = Path(weights_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"student_epoch_{epoch + 1}.pt"
        destination_tmp = destination.with_suffix(".pt.tmp")
        payload = {
            "epoch": epoch + 1,
            "student_model_name": self.config.student_model_name,
            "teacher_model_name": self.config.teacher_model_name,
            "model_state_dict": self.model_student.state_dict(),
        }

        local_dir = Path(self.config.save_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, local_tmp_name = tempfile.mkstemp(
            prefix=f".student_epoch_{epoch + 1}_",
            suffix=".pt",
            dir=local_dir,
        )
        os.close(file_descriptor)
        local_tmp = Path(local_tmp_name)

        try:
            torch.save(payload, local_tmp)
            shutil.copy2(local_tmp, destination_tmp)
            os.replace(destination_tmp, destination)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise OSError(
                    f"Saved student weights are missing or empty: {destination}"
                )
        finally:
            local_tmp.unlink(missing_ok=True)
            destination_tmp.unlink(missing_ok=True)

        print(f"Student weights saved: {destination}")

    @staticmethod
    def _benchmark_name(path: str, split: str) -> str:
        name = Path(path).stem
        suffix = f"_{split}"
        return name.removesuffix(suffix)

    @staticmethod
    def _metric_details(values: dict[str, Any]) -> str:
        labels = {
            "accuracy": "Acc",
            "f1": "F1",
            "precision": "P",
            "recall": "R",
            "average_precision": "AP",
            "spearman": "Spearman",
        }
        details = []
        for key, label in labels.items():
            value = values.get(key)
            if isinstance(value, (int, float)):
                details.append(f"{label}={100.0 * float(value):.2f}")
        return " ".join(details)

    @staticmethod
    def _benchmark_group_averages(
        scores_by_benchmark: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """Average the primary scores over IOD, OOD and all benchmarks.

        Every benchmark contributes its own primary metric (macro-F1, AP or
        Spearman), all on a 0-1 scale, so the groups are unweighted means over
        benchmarks rather than over examples.
        """
        groups = {
            "avg_iod": (
                "AVG (IOD)",
                sorted(name for name in scores_by_benchmark if name in IOD_BENCHMARKS),
            ),
            "avg_ood": (
                "AVG (OOD)",
                sorted(
                    name for name in scores_by_benchmark if name not in IOD_BENCHMARKS
                ),
            ),
            "avg_all": ("AVG (ALL)", sorted(scores_by_benchmark)),
        }
        return {
            key: {
                "label": label,
                "score": sum(scores_by_benchmark[name] for name in members)
                / len(members),
                "members": members,
            }
            for key, (label, members) in groups.items()
            if members
        }

    def print_evaluation_table(
        self,
        split: str,
        results: dict[str, Any],
        final: bool = False,
    ) -> None:
        primary_metrics = {
            "classification": "f1",
            "pair": "average_precision",
            "sts": "spearman",
        }
        rows = []
        scores_by_benchmark: dict[str, float] = {}
        for family in ("classification", "pair", "sts"):
            for path, raw_values in results.get(family, {}).items():
                values = (
                    {"spearman": raw_values} if family == "sts" else dict(raw_values)
                )
                metric_name = primary_metrics[family]
                score = float(values[metric_name])
                benchmark = self._benchmark_name(path, split)
                scores_by_benchmark[benchmark] = score
                rows.append(
                    (
                        family,
                        benchmark,
                        metric_name,
                        f"{100.0 * score:.2f}",
                        self._metric_details(values),
                    )
                )

        averages = self._benchmark_group_averages(scores_by_benchmark)
        divider_index = len(rows)
        for group in averages.values():
            rows.append(
                (
                    "summary",
                    group["label"],
                    "mean",
                    f"{100.0 * group['score']:.2f}",
                    " ".join(group["members"]),
                )
            )

        if split == "validation":
            title = f"VALIDATION - EPOCH {self.current_epoch + 1}"
        elif final:
            title = "FINAL TEST"
        else:
            title = f"TEST - EPOCH {self.current_epoch + 1}"
        if split == "test" and results.get("pair_threshold_source") == "test":
            title += "  (pair thresholds calibrated on the test split)"
        headers = ("Family", "Benchmark", "Primary metric", "Score", "Details")
        widths = [
            max([len(headers[index]), *(len(row[index]) for row in rows)])
            for index in range(len(headers))
        ]
        separator = "-+-".join("-" * width for width in widths)

        print("\n" + "=" * len(separator))
        print(title)
        print("=" * len(separator))
        print(
            " | ".join(
                headers[index].ljust(widths[index]) for index in range(len(headers))
            )
        )
        print(separator)
        for position, row in enumerate(rows):
            if position == divider_index:
                print(separator)
            print(
                " | ".join(row[index].ljust(widths[index]) for index in range(len(row)))
            )
        print("=" * len(separator) + "\n")

        return {key: group["score"] for key, group in averages.items()}

    def evaluate(self, split: str = "validation", final: bool = False):
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        threshold_source = getattr(self.config, "pair_threshold_source", "validation")
        if threshold_source not in {"validation", "test"}:
            raise ValueError(
                f"Unsupported pair_threshold_source={threshold_source!r}; "
                "expected 'validation' or 'test'"
            )
        if split == "validation":
            classification_tasks = eval_cls_tasks
            pair_tasks = eval_pair_tasks
            sts_tasks = eval_sts_tasks
            thresholds = None
        else:
            classification_tasks = test_cls_tasks
            pair_tasks = test_pair_tasks
            sts_tasks = test_sts_tasks
            if threshold_source == "test":
                # Threshold swept on the same split it is scored on. The pair
                # accuracy/F1/precision/recall then read as an upper bound, not a
                # held-out estimate; average_precision is unaffected either way.
                thresholds = None
            else:
                thresholds = getattr(self, "pair_validation_thresholds", None)
                if thresholds is None:
                    raise RuntimeError(
                        "Pair test evaluation requires thresholds selected on "
                        "validation data. Run validation first, or set "
                        "pair_threshold_source='test' to calibrate on the test split."
                    )

        student_model = self.model_student
        classification = eval_classification_task(
            student_model, classification_tasks, self.tok_student
        )
        pair, selected_thresholds = eval_pair_task(
            student_model,
            pair_tasks,
            self.tok_student,
            thresholds=thresholds,
        )
        sts = eval_sts_task(student_model, sts_tasks, self.tok_student)
        if split == "validation":
            self.pair_validation_thresholds = selected_thresholds
        results = {
            "classification": classification,
            "pair": pair,
            "sts": sts,
            # Recorded per evaluation so a later reader of metrics.jsonl can tell a
            # held-out pair score from a self-calibrated one.
            "pair_threshold_source": "test" if thresholds is None else "validation",
        }
        results["summary"] = self.print_evaluation_table(split, results, final=final)
        return results

    def log_step_records(self, records: list[dict]):
        """Append one JSONL line per training step to `step_metrics.jsonl`.

        Written separately from metrics.jsonl, which stays one record per epoch: the
        two have different row counts and different consumers, and mixing them would
        force every reader of the epoch table to filter.
        """
        if not records or not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "step_metrics.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(record, default=float, sort_keys=True) + "\n"
                for record in records
            )

    def log_depth_records(self) -> dict[str, Any] | None:
        """Flush the epoch's per-depth samples to `depth_metrics.jsonl` and summarise.

        Kept out of step_metrics.jsonl on purpose: these rows are per-layer curves
        sampled every `depth_log_every` steps, so they have a different shape and a
        different cadence from the per-step scalars.
        """
        records, self._depth_records = self._depth_records, []
        if not records:
            return None

        if self.config.save_dir:
            os.makedirs(self.config.save_dir, exist_ok=True)
            path = os.path.join(self.config.save_dir, "depth_metrics.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.writelines(
                    json.dumps(record, default=float, sort_keys=True) + "\n"
                    for record in records
                )

        curves = (
            "cos_teacher",
            "gram_gap",
            "energy",
            "dyn_residual",
            "field_norm",
            "step_norm",
            "direction_alignment",
        )
        skip = {"method", "epoch", "global_step", "step", "batch_size", "layers"}
        summary: dict[str, Any] = {"samples": len(records)}
        for key in curves:
            columns = zip(*(record[key] for record in records))
            summary[key] = [sum(column) / len(records) for column in columns]
        for key, value in records[0].items():
            if key in skip or key in curves or not isinstance(value, (int, float)):
                continue
            summary[key] = sum(record[key] for record in records) / len(records)

        self.print_depth_profile(summary)
        return summary

    @staticmethod
    def print_depth_profile(summary: dict[str, Any]) -> None:
        """Print the depth profile the paper's hypotheses are stated over."""
        headers = (
            "layer",
            "cos(teacher)",
            "gram gap",
            "energy",
            "dyn resid",
            "align",
            "|dz|",
            "|dt*F|",
        )
        cosines = summary["cos_teacher"]
        rows = []
        for index, cosine in enumerate(cosines):
            transition = index if index < len(summary["dyn_residual"]) else None
            rows.append(
                (
                    str(index + 1),
                    f"{cosine:.4f}",
                    f"{summary['gram_gap'][index]:.4f}",
                    f"{summary['energy'][index]:.4f}",
                    "-"
                    if transition is None
                    else f"{summary['dyn_residual'][transition]:.4f}",
                    "-"
                    if transition is None
                    else f"{summary['direction_alignment'][transition]:+.3f}",
                    "-"
                    if transition is None
                    else f"{summary['step_norm'][transition]:.4f}",
                    "-"
                    if transition is None
                    else f"{summary['field_norm'][transition]:.5f}",
                )
            )

        widths = [
            max([len(headers[i]), *(len(row[i]) for row in rows)])
            for i in range(len(headers))
        ]
        separator = "-+-".join("-" * width for width in widths)
        print(f"\nDepth profile (mean over {summary['samples']} sampled batches)")
        print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        print(separator)
        for row in rows:
            print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
        # The rows are the curves; these are the claims the curves are supposed to
        # support, so they are printed next to them rather than left to the reader.
        print(
            f"cos {summary['cos_first']:.4f} -> {summary['cos_final']:.4f} "
            f"(gain {summary['cos_gain']:+.4f}, non-monotone at "
            f"{summary['cos_violations']:.1f}/{len(cosines) - 1} depths, "
            f"curvature {summary['cos_curvature']:.4f})"
        )
        print(
            f"gram gap {summary['gram_gap_first']:.4f} -> "
            f"{summary['gram_gap_final']:.4f} "
            f"(contraction {summary['gram_gap_contraction']:+.4f}, non-monotone at "
            f"{summary['gram_violations']:.1f}/{len(cosines) - 1} depths)"
        )
        print(
            f"energy {summary['energy_first']:.4f} -> {summary['energy_final']:.4f} "
            f"(rises at {summary['energy_violations']:.1f}/{len(cosines) - 1} depths; "
            "Prop. 2 forbids this for the ideal flow)"
        )
        print(
            f"mean alignment {summary['mean_alignment']:+.3f}  "
            f"anisotropy student {summary['student_anisotropy']:.4f} vs teacher "
            f"{summary['teacher_anisotropy']:.4f}\n"
        )

    def log_experiment_record(self, record: dict[str, Any]):
        if not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "metrics.jsonl")
        payload = {
            "method": self.config.distill_method,
            "seed": self.config.seed,
            **record,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=float, sort_keys=True) + "\n")

    def train(self):
        cfg = self.config

        if cfg.distill_method == "stella":
            print("\n" + "=" * 70)
            print("Starting Stella 2-Stage Training...")
            print("=" * 70)
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Stage 1 epochs: {cfg.epochs_stage1}")
            print(f"Stage 2 epochs: {cfg.epochs_stage2}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("=" * 70 + "\n")

            print("\n" + "=" * 70)
            print("STAGE 1: Freeze backbone + fc2,3,4, train fc1 only")
            print("=" * 70)

            for p in self.model_student.backbone.parameters():
                p.requires_grad = False
            for head in [
                self.model_student.fc2,
                self.model_student.fc3,
                self.model_student.fc4,
            ]:
                for p in head.parameters():
                    p.requires_grad = False

            print("Frozen: backbone, fc2, fc3, fc4")
            print("Trainable: fc1")

            self.current_stage = 1
            for epoch in range(cfg.epochs_stage1):
                avg_loss = self.train_epoch(epoch)
                self.log_experiment_record(
                    {"stage": 1, "train": self.last_epoch_metrics}
                )

                if should_save_epoch(epoch, cfg.save_every):
                    self.save_checkpoint(epoch, {"loss": avg_loss})

            print("\n" + "=" * 70)
            print("STAGE 1 COMPLETED!")
            print("=" * 70 + "\n")

            print("\n" + "=" * 70)
            print("STAGE 2: Unfreeze all, train full model")
            print("=" * 70)

            for p in self.model_student.parameters():
                p.requires_grad = True

            print("Unfrozen: all parameters")
            print("Trainable: backbone, fc1, fc2, fc3, fc4")

            self.optimizer = optim.AdamW(
                self.model_student.parameters(), lr=cfg.learning_rate
            )
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(len(self.train_loader) * cfg.warmup_ratio),
                num_training_steps=len(self.train_loader) * cfg.epochs_stage2,
            )

            self.step_times = []
            self.ma_window = deque(maxlen=50)

            self.current_stage = 2
            for epoch in range(cfg.epochs_stage2):
                avg_loss = self.train_epoch(epoch)
                validation_results = None

                print("\n" + "=" * 60)
                print(f"Evaluation after Stage2 Epoch {epoch + 1}")
                print("=" * 60)

                stage2_split = (
                    "test" if cfg.evaluate_test_each_epoch else "validation"
                )
                try:
                    validation_results = self.evaluate(stage2_split)
                except Exception as e:
                    print(f"Warning: Evaluation failed with error: {e}")
                    print("Continuing training...")

                print("=" * 60 + "\n")
                self.log_experiment_record(
                    {
                        "stage": 2,
                        "train": self.last_epoch_metrics,
                        stage2_split: validation_results,
                    }
                )

                if should_save_epoch(epoch, cfg.save_every):
                    self.save_checkpoint(epoch, {"loss": avg_loss})

            print("\n" + "=" * 70)
            print("STAGE 2 COMPLETED!")
            print("=" * 70)

            self.save_checkpoint(cfg.epochs_stage2 - 1, {"loss": avg_loss})
            try:
                test_results = self.evaluate("test", final=True)
                if (
                    getattr(self, "use_wandb", False)
                    and WANDB_AVAILABLE
                    and test_results is not None
                ):
                    wandb.log(
                        self._flatten_metrics("test", test_results),
                        step=self.global_step,
                    )
                self.log_experiment_record({"stage": 2, "test": test_results})
            except Exception as e:
                print(f"Warning: Final test evaluation failed with error: {e}")

            print("\n" + "=" * 70)
            print("Training completed successfully!")
            print("=" * 70)

        else:
            print("\n" + "=" * 60)
            print("Starting training...")
            print("=" * 60)
            print(f"Method: {cfg.distill_method}")
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Epochs: {cfg.epochs}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("=" * 60 + "\n")

            eval_split = "test" if cfg.evaluate_test_each_epoch else "validation"
            epoch_results = None

            for epoch in range(cfg.epochs):
                avg_loss = self.train_epoch(epoch)
                epoch_results = None

                refit_every = int(getattr(cfg, "gauge_refit_every", 0) or 0)
                if (
                    refit_every > 0
                    and (epoch + 1) % refit_every == 0
                    and epoch + 1 < cfg.epochs
                ):
                    self._refit_gauge(epoch)

                print("\n" + "=" * 60)
                print(f"Evaluation after Epoch {epoch + 1}")
                print("=" * 60)

                if cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
                    try:
                        epoch_results = self.evaluate(eval_split)
                        if (
                            getattr(self, "use_wandb", False)
                            and WANDB_AVAILABLE
                            and epoch_results is not None
                        ):
                            wandb.log(
                                self._flatten_metrics(eval_split, epoch_results),
                                step=self.global_step,
                            )
                    except Exception as e:
                        print(f"Warning: {eval_split} evaluation failed with error: {e}")
                        print("Continuing training...")

                print("=" * 60 + "\n")
                # Keyed by the split that actually ran, so a reader of metrics.jsonl
                # never has to guess which data a per-epoch number came from.
                self.log_experiment_record(
                    {
                        "train": self.last_epoch_metrics,
                        eval_split: epoch_results,
                    }
                )

                if should_save_epoch(epoch, cfg.save_every):
                    try:
                        self.save_checkpoint(epoch, {"loss": avg_loss})
                    except Exception as e:
                        if getattr(cfg, "weights_dir", None):
                            raise RuntimeError(
                                f"Required epoch {epoch + 1} weights could not be saved"
                            ) from e
                        print(f"Warning: Saving checkpoint failed with error: {e}")
                        print("Continuing training...")

            print("\n" + "=" * 60)
            print("Training completed!")
            print("=" * 60)
            print("Done train()")

            self.save_checkpoint(cfg.epochs - 1, {"loss": avg_loss})
            try:
                # The last epoch already scored the test split under this setting, on
                # the same weights: re-running it would only burn a second pass.
                reusable = (
                    eval_split == "test"
                    and epoch_results is not None
                    and cfg.eval_every
                    and cfg.epochs % cfg.eval_every == 0
                )
                if reusable:
                    print("Reusing the final epoch's test evaluation")
                    test_results = epoch_results
                    self.print_evaluation_table("test", test_results, final=True)
                else:
                    test_results = self.evaluate("test", final=True)
                if (
                    getattr(self, "use_wandb", False)
                    and WANDB_AVAILABLE
                    and test_results is not None
                ):
                    wandb.log(
                        self._flatten_metrics("test", test_results),
                        step=self.global_step,
                    )
                self.log_experiment_record({"test": test_results})
            except Exception as e:
                print(f"Warning: Final test evaluation failed with error: {e}")

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
