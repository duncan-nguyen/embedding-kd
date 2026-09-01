import json
import os
import random
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_scheduler
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
from src.cache_teacher import (
    cache_filename,
    cache_teacher_embeddings,
    corpus_digest,
    load_cached_embeddings,
    validate_cached_embeddings,
)
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.geoode_kd import GeoODEKD
from src.criterions.relational_kd import RelationalKD
from src.criterions.simcse import SimCSEOnly
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
from src.evaluation.retrieval import eval_retrieval_task, test_retrieval_tasks
from src.loss import info_nce
from src.pooling import mean_pooling, pool_sentence_embedding
from src.target_projector import LearnedTargetProjector
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_gauge_rotation,
    fit_teacher_projection,
    project_teacher_embeddings,
    retained_energy,
)

# projection_type values whose map is trained rather than fitted, and the direction
# each one hands to LearnedTargetProjector.
LEARNED_PROJECTIONS = {"learned_t2s": "t2s", "learned_s2t": "s2t"}

# The distillation corpus is drawn from EMOTION, WiC and STS-B, so those three
# benchmarks are in-distribution and the remaining ones are held out. Reporting
# them as one number would let an in-distribution gain stand in for transfer, so
# the table averages them apart.
IOD_BENCHMARKS = frozenset({"emotion", "wic", "stsb"})
# Scored by nDCG@10 over a whole corpus rather than over a sentence pair, so
# they get their own summary row instead of diluting the sentence-level AVG
# (OOD) that earlier runs are reported against.
RETRIEVAL_BENCHMARKS = frozenset({"arguana", "fiqa", "scidocs"})


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


class StepTimer:
    """Per-step duration measured on the CUDA stream instead of by stalling on it.

    The straightforward way to time a step -- ``synchronize(); t0 = perf_counter();
    step(); synchronize()`` -- measures the right thing and costs the pipeline the
    very overlap it is there to report. Both synchronisations drain the whole queued
    backlog, so while they run the DataLoader cannot prefetch, the next batch cannot
    start its host-to-device copy, and the CPU cannot enqueue anything: on a small
    student the step is a few milliseconds and the stall is a visible fraction of it.

    A pair of CUDA events records the same interval as timestamps *inside* the
    stream, so the gap they measure still includes any time the GPU spent idle
    waiting for the CPU. The reading is simply taken later: each iteration collects
    whichever earlier steps have since completed (``query()`` never blocks), and
    :meth:`finish` drains the rest at the end of the epoch, where one
    synchronisation costs nothing. Durations come back in step order either way.

    Events are recorded on the current device. For the methods that run a teacher on
    a second GPU the teacher's kernels are enqueued from the same host thread and
    the student's work depends on their output, so they serialise into this interval
    rather than hiding from it.
    """

    def __init__(self) -> None:
        self.cuda = torch.cuda.is_available()
        self._pending: deque = deque()
        self._seconds: list[float] = []
        self._read = 0
        self._start = None

    def start(self) -> None:
        if self.cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.perf_counter()

    def stop(self) -> None:
        if self._start is None:
            raise RuntimeError("StepTimer.stop() without a matching start()")
        if self.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((self._start, end))
            self._collect()
        else:
            self._seconds.append(time.perf_counter() - self._start)
        self._start = None

    def _collect(self, block: bool = False) -> None:
        while self._pending:
            start, end = self._pending[0]
            # query() is a non-blocking "has the GPU reached this marker yet".
            if not block and not end.query():
                break
            end.synchronize()
            self._seconds.append(start.elapsed_time(end) / 1000.0)
            self._pending.popleft()

    def take_new(self) -> list[float]:
        """Durations that have completed since the last call, in step order."""
        fresh = self._seconds[self._read :]
        self._read = len(self._seconds)
        return fresh

    def finish(self) -> list[float]:
        """Every step's duration, waiting for the stragglers. One synchronisation."""
        self._collect(block=True)
        self._read = len(self._seconds)
        return self._seconds


class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        self._validate_eval_config(config)
        self.wandb_run = None
        self.global_step = 0
        self.current_epoch = 0
        self.current_step = 0
        self._saved_checkpoint_epochs = set()
        # Set before setup_training, which fills it in for the methods that need it.
        self.proj_s2t = None
        self.setup_seed(config.seed)
        self.setup_devices()
        self.setup_models()
        self.setup_data()
        self.setup_training()
        self.setup_wandb()
        self.criterion = self._build_criterion()

        # Metrics tracking
        self.step_times = []
        self.ma_window = deque(maxlen=50)
        self.warmup_steps = 10

    # ------------------------------------------------------------------ criterion

    def _build_criterion(self):
        """The objective of the selected method.

        ``None`` for TALAS: its criterion sizes itself from the first batch's layer
        count, so it is built in the training step instead (together with the SAM
        optimizer that owns its parameters).
        """
        builders = {
            "cdm": self._build_cdm_criterion,
            "dskd": self._build_dskd_criterion,
            "emo": self._build_emo_criterion,
            "geoode": self._build_geoode_criterion,
            "rkd": self._build_rkd_criterion,
            "simcse": self._build_simcse_criterion,
        }
        builder = builders.get(self.config.distill_method)
        return None if builder is None else builder()

    def _add_criterion_to_optimizer(self, criterion, lr_scale: float = 1.0) -> None:
        """Give a criterion's own parameters an optimizer group and rebuild the
        scheduler.

        The rebuild is not optional: a group added after the scheduler was
        constructed has no matching ``base_lr``, and ``scheduler.step()`` then fails
        on the length mismatch.
        """
        self.optimizer.add_param_group(
            {
                "params": criterion.parameters(),
                "lr": self.config.learning_rate * lr_scale,
            }
        )
        self.scheduler = self._build_scheduler()

    def _build_cdm_criterion(self):
        cfg = self.config
        return ContextualDynamicMapping(
            tok_student=self.tok_student,
            tok_teacher=self.tok_teacher,
            blending_model_special_token=cfg.teacher_special_token,
            base_model_special_token=cfg.student_special_token,
            w_task=cfg.w_task,
            alpha_dtw=cfg.alpha_dtw,
            debug_align=cfg.debug_align,
        )

    def _build_dskd_criterion(self):
        cfg = self.config
        criterion = DualSpaceKD(
            student_dim=self.model_student.config.hidden_size,
            teacher_dim=self.model_teacher.config.hidden_size,
            w_task=cfg.w_task,
            alpha_dtw=cfg.alpha_dtw,
        ).to(self.device_s)
        self._add_criterion_to_optimizer(criterion)
        print("DSKD criterion initialized and added to optimizer")
        return criterion

    def _build_emo_criterion(self):
        cfg = self.config
        criterion = EMODistillation(
            d_teacher=self.model_teacher.config.hidden_size,
            d_student=self.model_student.config.hidden_size,
            k_layers=getattr(cfg, "k_layers", 1),
            alpha_ot=getattr(cfg, "alpha_ot", 0.1),
            max_iter=getattr(cfg, "max_iter_ot", 100),
            teacher_special=getattr(cfg, "teacher_special_token", "<s>"),
            student_special=getattr(cfg, "student_special_token", "[CLS]"),
        ).to(self.device_s)
        self._add_criterion_to_optimizer(criterion)
        print("EMO criterion initialized and added to optimizer")
        return criterion

    def _build_geoode_criterion(self):
        # GeoODE-KD holds no parameters of its own: the targets are fitted and
        # frozen before training, so nothing is added to the optimizer and the
        # deployed student is the unmodified encoder. The learned-projector
        # baselines are the exception, and the only one -- they put a trainable
        # linear map where the frozen P_T would be, which is the thing under test.
        cfg = self.config
        target_projector = None
        learned = getattr(self, "_learned_projector", None)
        if learned is not None:
            target_projector = LearnedTargetProjector(
                teacher_dim=learned["teacher_dim"],
                student_dim=learned["student_dim"],
                direction=learned["direction"],
                eps=cfg.eps_norm,
            )
        criterion = GeoODEKD(
            lambda_end=cfg.lambda_end,
            lambda_ctr=cfg.lambda_ctr,
            contrastive_temperature=cfg.contrastive_temperature,
            endpoint_loss=getattr(cfg, "endpoint_loss", "cosine"),
            lambda_gram=float(getattr(cfg, "lambda_gram", 0.0) or 0.0),
            lambda_topo=float(getattr(cfg, "lambda_topo", 0.0) or 0.0),
            topo_metric=getattr(cfg, "topo_metric", "chord"),
            pooling=cfg.student_pooling,
            include_embedding_layer=cfg.include_embedding_layer,
            eps_norm=cfg.eps_norm,
            target_projector=target_projector,
        ).to(self.device_s)
        if target_projector is not None:
            # GeoODEKD owns no other parameters, so this param group is exactly
            # the projector.
            scale = float(getattr(cfg, "learned_projector_lr_scale", 1.0))
            self._add_criterion_to_optimizer(criterion, lr_scale=scale)
            trainable = sum(p.numel() for p in criterion.parameters())
            print(
                f"Learned target map added to the optimizer: {target_projector} "
                f"({trainable:,} parameters, lr x{scale}). Training only -- "
                "inference is still the plain student encoder"
            )
        print(
            "GeoODE-KD criterion initialized: "
            f"lambda_end={cfg.lambda_end}, lambda_ctr={cfg.lambda_ctr}, "
            f"endpoint_loss={getattr(cfg, 'endpoint_loss', 'cosine')}, "
            f"lambda_gram={float(getattr(cfg, 'lambda_gram', 0.0) or 0.0)}, "
            f"lambda_topo={float(getattr(cfg, 'lambda_topo', 0.0) or 0.0)} "
            f"({getattr(cfg, 'topo_metric', 'chord')})"
        )
        return criterion

    def _build_rkd_criterion(self):
        # RKD holds no parameters either: both of its potentials are invariant
        # to the width of the space, so the teacher supervises the student
        # across the dimensionality gap with nothing fitted in between.
        cfg = self.config
        criterion = RelationalKD(
            w_task=cfg.w_task,
            w_dist=cfg.w_dist,
            w_angle=cfg.w_angle,
            huber_delta=cfg.huber_delta,
            normalize_student=cfg.normalize_student,
            eps=cfg.eps_norm,
        ).to(self.device_s)
        print(
            "RKD criterion initialized: "
            f"w_task={cfg.w_task}, w_dist={cfg.w_dist}, "
            f"w_angle={cfg.w_angle}, "
            f"normalize_student={cfg.normalize_student}"
        )
        return criterion

    def _build_simcse_criterion(self):
        cfg = self.config
        criterion = SimCSEOnly(temperature=cfg.temperature).to(self.device_s)
        print(
            "SimCSE-only control initialized: "
            f"view={cfg.simcse_view}, temperature={cfg.temperature}. "
            "No teacher term is in this objective."
        )
        return criterion

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

    @staticmethod
    def _embedding_dim_of(model_config, model_name: str) -> int:
        """Hidden width of a model config, looking through a wrapped text_config."""
        dim = getattr(model_config, "hidden_size", None)
        if dim is None:
            text_config = getattr(model_config, "text_config", None)
            dim = getattr(text_config, "hidden_size", None)
        if dim is None:
            raise ValueError(
                f"Could not read an embedding dim from the config of {model_name}"
            )
        return int(dim)

    def _resolve_teacher_embedding_dim(self) -> int:
        """Embedding width of the teacher, read from its config without loading it.

        The teacher weights are loaded further down, but the Stella student's fc1
        has to be built before that, so the width comes from the config file
        instead of from a materialized model.
        """
        teacher_config = AutoConfig.from_pretrained(
            self.config.teacher_model_name,
            trust_remote_code=True,
        )
        try:
            return self._embedding_dim_of(
                teacher_config, self.config.teacher_model_name
            )
        except ValueError as error:
            raise ValueError(f"{error}; set output_dim1 to it manually.") from error

    def _pool_teacher(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Sentence vector of the online teacher under ``pooling_method``.

        The same dispatch the cached-teacher methods run at cache time, so
        ``--teacher_pooling`` means one thing for every method: a BGE-style encoder
        is read at its CLS position, a Qwen3-style decoder at its last token.
        """
        return pool_sentence_embedding(
            last_hidden_state,
            attention_mask,
            getattr(self.config, "pooling_method", "last_token"),
        )

    def setup_models(self):
        cfg = self.config

        print("Loading tokenizers...")
        tokenizer_kwargs = {"use_fast": True}
        self.tok_student = AutoTokenizer.from_pretrained(
            cfg.student_model_name,
            **tokenizer_kwargs,
        )
        # The SimCSE-only control has no teacher term, so neither the teacher
        # tokenizer nor the teacher weights are loaded: the run is the student's own
        # contrastive objective on the same corpus and nothing else.
        self.tok_teacher = None
        if cfg.distill_method != "simcse":
            self.tok_teacher = AutoTokenizer.from_pretrained(
                cfg.teacher_model_name,
                trust_remote_code=True,
                **tokenizer_kwargs,
            )
        # transformers >= 5 renamed the loading keyword and, by default, loads a
        # checkpoint in the dtype it was saved in. A student saved in fp16 (the
        # jim12345 MiniLMv2 checkpoints are) would then train in fp16, and the
        # GradScaler refuses to unscale fp16 gradients. The student is therefore
        # loaded in fp32 unless the config asks for something else; mixed precision
        # comes from autocast, not from the parameter dtype.
        try:
            transformers_major = int(transformers_version.split(".", maxsplit=1)[0])
        except (TypeError, ValueError):
            transformers_major = 4
        dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
        model_dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        student_dtype_name = getattr(cfg, "student_dtype", None) or "float32"
        if student_dtype_name not in model_dtypes:
            raise ValueError(
                f"Unsupported student_dtype={student_dtype_name!r}; "
                f"expected one of {sorted(model_dtypes)}"
            )
        student_torch_dtype = model_dtypes[student_dtype_name]

        if cfg.distill_method == "stella":
            print(f"Loading Stella student model: {cfg.student_model_name}")
            # Stage 1 and stage 2 both take a cosine loss between fc1 and the
            # teacher embedding itself, so fc1 has to land in the teacher's
            # dimension: one head per teacher, sized to that teacher's vector.
            # The default (1024) only happens to fit Qwen3-Embedding-0.6B; a
            # larger teacher moves it. fc2-fc4 are compared to fc1 through Gram
            # matrices only, so those Matryoshka dims stay free.
            teacher_dim = self._resolve_teacher_embedding_dim()
            configured_dim = getattr(cfg, "output_dim1", 1024)
            if configured_dim != teacher_dim:
                print(
                    f"[stella] output_dim1={configured_dim} does not match the "
                    f"{cfg.teacher_model_name} embedding dim ({teacher_dim}); "
                    f"sizing fc1 to {teacher_dim}."
                )
                cfg.output_dim1 = teacher_dim
            self.model_student = StellaModel(
                cfg.student_model_name,
                output_dim1=teacher_dim,
                pooling=getattr(cfg, "pooling", "cls"),
                output_dim2=getattr(cfg, "output_dim2", 512),
                output_dim3=getattr(cfg, "output_dim3", 256),
                output_dim4=getattr(cfg, "output_dim4", 128),
                backbone_kwargs={dtype_argument: student_torch_dtype},
            )
            self.current_stage = 1
        else:
            print(f"Loading student model: {cfg.student_model_name}")
            student_kwargs = {dtype_argument: student_torch_dtype}

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

        self.model_teacher = None
        if cfg.distill_method == "simcse":
            print(
                "SimCSE-only: no teacher model is loaded "
                f"(the control for {cfg.teacher_model_name})"
            )
        else:
            print(f"Loading teacher model: {cfg.teacher_model_name}")
            teacher_kwargs = {"trust_remote_code": True}
            if cfg.teacher_dtype in ("bfloat16", "float16"):
                teacher_kwargs[dtype_argument] = model_dtypes[cfg.teacher_dtype]

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
            print(
                f"Teacher pooling: {getattr(cfg, 'pooling_method', 'last_token')} "
                f"(dim {self._embedding_dim_of(self.model_teacher.config, cfg.teacher_model_name)})"
            )

        self.model_student.to(self.device_s)

        student_dtype = next(self.model_student.parameters()).dtype
        print(f"Student training dtype: {student_dtype}")
        if student_dtype != student_torch_dtype:
            raise RuntimeError(
                f"Student loaded as {student_dtype} although {student_torch_dtype} "
                "was requested; the checkpoint dtype leaked through the loader"
            )
        assert_module_parameters_finite(self.model_student, "Student model after load")

        if self.model_teacher is not None:
            self.model_teacher.to(self.device_t)
            self.model_teacher.eval()
            for p in self.model_teacher.parameters():
                p.requires_grad_(False)

        print("Models loaded successfully!")
        print("Done setup_models")

    def setup_data(self):
        cfg = self.config

        print(f"Loading training data from: {cfg.train_data_path}")
        df = pd.read_csv(cfg.train_data_path)

        if cfg.task_type == "pair_cls" and (
            "premise" not in df.columns or "hypothesis" not in df.columns
        ):
            # A raw one-column corpus is read as a degenerate pair, so the two
            # sides of the contrastive term are the same sentence under dropout.
            text = df["text"] if "text" in df.columns else df.iloc[:, 0]
            df["premise"] = text
            df["hypothesis"] = text

        self.task_head = self._build_task_head(df)

        # TALAS, GeoODE-KD and RKD all train against cached teacher embeddings only:
        # the teacher is run once, offline, and never during student optimization.
        if cfg.distill_method in ("talas", "geoode", "rkd"):
            self._setup_cached_teacher_data(df)
        else:
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

    def _build_task_head(self, df: pd.DataFrame) -> nn.Module | None:
        """The supervised head EMO trains next to its KD terms, when the corpus
        carries labels. Every other method's task term is unsupervised."""
        cfg = self.config
        if cfg.distill_method != "emo" or "label" not in df.columns:
            return None
        hidden_size = self.model_student.config.hidden_size
        num_labels = int(df["label"].nunique())
        if cfg.task_type == "single_cls":
            return nn.Linear(hidden_size, num_labels).to(self.device_s)
        if cfg.task_type == "pair_cls":
            # [u, v, |u - v|, u * v], the sentence-pair feature of InferSent/SBERT.
            return nn.Linear(hidden_size * 4, num_labels).to(self.device_s)
        return None

    def _teacher_cache(self, df: pd.DataFrame) -> torch.Tensor:
        """The teacher's embedding of every training row, loaded or computed once.

        Identity of the signal is checked on load: the corpus enters by its
        *contents*, since a file rebuilt to the same path with the same row count
        would pass every other check.
        """
        cfg = self.config
        digest = corpus_digest(cfg.train_data_path)
        cache_path = self._resolve_cache_path(digest)
        teacher_dim = self._embedding_dim_of(
            self.model_teacher.config, cfg.teacher_model_name
        )

        if cache_path.exists():
            print(f"Loading cached teacher embeddings from: {cache_path}")
            teacher_cls, cached_metadata = load_cached_embeddings(str(cache_path))
            # The cache is keyed by its path alone, so this is the only place a
            # teacher/pooling swap behind an unchanged --cache_path gets caught.
            validate_cached_embeddings(
                teacher_cls,
                cached_metadata,
                str(cache_path),
                teacher_model_name=cfg.teacher_model_name,
                pooling_method=cfg.pooling_method,
                normalize=bool(cfg.normalize_cache),
                teacher_dim=teacher_dim,
                rows=len(df),
                max_length=int(cfg.max_length),
                train_data_digest=digest,
            )
            print(
                f"Loaded {len(teacher_cls)} cached embeddings "
                "(teacher not run for this training)"
            )
            self._check_cache_rows(teacher_cls, df, cache_path)
            return teacher_cls

        print("Cache not found. Pre-computing teacher embeddings...")
        os.makedirs(cache_path.parent, exist_ok=True)
        teacher_cls = cache_teacher_embeddings(
            model_teacher=self.model_teacher,
            texts=self._cache_texts(df),
            tokenizer=self.tok_teacher,
            device=self.device_t,
            max_length=int(cfg.max_length),
            # Forward-only and under inference mode, so the batch that fits is much
            # larger than the training batch, which is what cfg.batch_size sizes.
            batch_size=self._cache_batch_size(),
            pooling_method=cfg.pooling_method,
            normalize=cfg.normalize_cache,
            dtype=torch.float32 if cfg.cache_dtype == "float32" else torch.float16,
            cache_path=str(cache_path),
            metadata={
                "teacher_model_name": cfg.teacher_model_name,
                "pooling_method": cfg.pooling_method,
                "normalize": bool(cfg.normalize_cache),
                "train_data_path": str(cfg.train_data_path),
                "train_data_digest": digest,
                "max_length": int(cfg.max_length),
            },
        )
        print(f"Cached {len(teacher_cls)} teacher embeddings to {cache_path}")
        self._check_cache_rows(teacher_cls, df, cache_path)
        return teacher_cls

    @staticmethod
    def _cache_texts(df: pd.DataFrame) -> list[str]:
        """The one string per corpus row that the teacher cache is built from.

        The cached methods supervise the student on the embedding of the row's
        *first* text, so that is the only column the teacher ever has to encode --
        for a one-column corpus read as a degenerate pair, the second column is a
        copy of it anyway.
        """
        if "premise" in df.columns:
            column = "premise"
        elif "sentence1" in df.columns:
            column = "sentence1"
        elif "text" in df.columns:
            column = "text"
        else:
            raise ValueError(
                "training data needs a 'premise', 'sentence1' or 'text' column to "
                f"build a teacher cache from; got {list(df.columns)}"
            )
        return df[column].astype(str).tolist()

    def _cache_batch_size(self) -> int:
        """Batch size of the teacher pass, which is not the training batch size.

        Nothing in this pass is kept for backward, so it fits a far larger batch
        than training does, and the batches are length-sorted so all but the last
        few are short. ``cache_batch_size = 0`` falls back to the training batch
        size for anyone who needs the old behaviour on a small card.
        """
        configured = int(getattr(self.config, "cache_batch_size", 0) or 0)
        return configured if configured > 0 else int(self.config.batch_size)

    @staticmethod
    def _check_cache_rows(
        teacher_cls: torch.Tensor, df: pd.DataFrame, cache_path: Path
    ) -> None:
        if len(teacher_cls) != len(df):
            raise ValueError(
                f"Cached teacher embeddings length mismatch: cache has "
                f"{len(teacher_cls)} rows but training data has {len(df)} rows. "
                f"Remove or regenerate {cache_path}."
            )

    def _setup_cached_teacher_data(self, df: pd.DataFrame) -> None:
        """Dataset and collate for the methods that read a frozen teacher cache.

        The teacher model is released here: once the cache exists it is never run
        again, and on a single-GPU box its weights are the memory the student needs.
        """
        cfg = self.config
        teacher_cls_list = self._teacher_cache(df)

        teacher_topo_list = None
        if cfg.distill_method == "geoode":
            if float(getattr(cfg, "lambda_topo", 0.0) or 0.0) > 0.0:
                # The H0 term compares point-cloud shapes, so it needs no shared
                # basis and reads the teacher *before* P_T narrows it to d_S --
                # the one supervision signal in the run that P_T cannot colour.
                teacher_topo_list = (
                    teacher_cls_list.clone().contiguous().share_memory_()
                )
            teacher_cls_list = self._project_teacher_targets(
                teacher_cls_list, df["premise"].astype(str).tolist()
            )
            # The gauge refit rewrites the targets in place between epochs; the
            # persistent DataLoader workers only see that through shared memory.
            teacher_cls_list = teacher_cls_list.contiguous().share_memory_()
        self.teacher_cls_all = teacher_cls_list

        del self.model_teacher
        self.model_teacher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Teacher model freed from GPU memory")

        self.train_ds = TextPairWithTeacher(
            df, cfg.task_type, teacher_cls_list, teacher_topo_list
        )
        self.collate_fn = DualTokenizerCollateWithTeacher(
            self.tok_student,
            cfg.task_type,
            cfg.max_length,
            need_second_text=self._needs_second_text(),
            # Only the token-level methods (cdm, dskd) align token strings, and none
            # of them reads a teacher cache.
            need_special_tokens_mask=False,
            topo_metric=(
                getattr(cfg, "topo_metric", "chord")
                if teacher_topo_list is not None
                else None
            ),
        )

    def _needs_second_text(self) -> bool:
        """Whether the second sentence of each row is read at all this run.

        Tokenising it, padding it, stacking it and copying it to the GPU is a per
        step cost, so it is worth knowing that GeoODE's default contrastive view --
        two dropout passes over the *first* sentence -- never touches it. TALAS and
        RKD do: their in-batch contrastive term takes the paired sentence as the
        positive.
        """
        cfg = self.config
        if cfg.task_type == "single_cls":
            return False
        if cfg.distill_method != "geoode":
            return True
        if float(getattr(cfg, "lambda_ctr", 0.0) or 0.0) <= 0.0:
            return False
        return getattr(cfg, "contrastive_view", "dropout") == "pair"

    def _resolve_cache_path(self, digest: str) -> Path:
        """Where this run's teacher cache lives.

        ``cache_dir`` is the reuse path: the filename is derived from what makes a
        cache reusable at all, so one directory can hold every cache a project
        builds and a run either finds exactly its own or misses. That is what
        ``cache_path`` cannot do -- a single name shared between runs of different
        pairs loads the wrong file and gets refused, and a name scoped to one run
        re-encodes the corpus every time.

        ``cache_path`` still wins when it was set explicitly, so an existing cache
        can always be pointed at directly.
        """
        cfg = self.config
        cache_dir = getattr(cfg, "cache_dir", None)
        if not cache_dir:
            return Path(cfg.cache_path)
        name = cache_filename(
            teacher_model_name=cfg.teacher_model_name,
            pooling_method=cfg.pooling_method,
            train_data_path=cfg.train_data_path,
            max_length=int(cfg.max_length),
            normalize=bool(cfg.normalize_cache),
            train_data_digest=digest,
        )
        path = Path(cache_dir) / name
        print(f"Teacher cache (shared directory): {path}")
        return path

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
            pooled = self._pool_student(last, encoded["attention_mask"])
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

        projection_type = getattr(cfg, "projection_type", "pca")
        if projection_type in LEARNED_PROJECTIONS:
            return self._learned_teacher_targets(teacher_cls, projection_type)

        projection, mean = fit_teacher_projection(
            teacher_cls,
            out_dim=student_dim,
            projection_type=projection_type,
            center=cfg.pca_center_fit,
            seed=int(getattr(cfg, "projection_seed", 0)),
        )
        # The MSE baseline (sentence-transformers recipe) regresses onto the raw
        # projected target, so it is the one case where norm(.) is skipped.
        renormalize = getattr(cfg, "endpoint_loss", "cosine") != "mse"
        targets = project_teacher_embeddings(
            teacher_cls,
            projection,
            mean=mean,
            subtract_mean=cfg.pca_subtract_mean,
            eps=cfg.eps_norm,
            renormalize=renormalize,
        )

        explained = 1.0
        if teacher_dim <= student_dim:
            print(
                f"Teacher dim {teacher_dim} <= student dim {student_dim}: "
                f"P_T discards nothing ({projection_type} map, targets are "
                "re-coordinatised and re-normalized)"
            )
        else:
            # Eckart-Young: among all rank-d_S linear maps, PCA retains the largest
            # share of the cached embedding energy, i.e. it is the linear map that
            # best preserves the teacher's Gram matrix. This number is
            # what the paper reports for P_T, and it is also the number the random
            # controls have to be read against: they span a d_S-subspace drawn
            # without looking at the teacher, so they retain about d_S/d_T.
            explained = retained_energy(teacher_cls, projection)
            print(
                f"Fitted {projection_type} teacher projection {teacher_dim} -> "
                f"{student_dim} (retains {explained:.1%} of cached embedding "
                f"energy; a random subspace retains ~{student_dim / teacher_dim:.1%})"
            )

        rotation = None
        gauge_stats = None
        gauge_mode = getattr(cfg, "gauge_rotation", "procrustes")
        refit_every = int(getattr(cfg, "gauge_refit_every", 0) or 0)
        if gauge_mode != "procrustes" and refit_every > 0:
            # The refit is an alternating exact minimisation over O(d_S) and is only
            # a descent step for the Procrustes gauge. Re-drawing a random rotation
            # every epoch is a different experiment; refusing the combination keeps
            # the two from being confused in the logs.
            raise ValueError(
                "gauge_refit_every > 0 is only defined for gauge_rotation="
                f"'procrustes'; the {gauge_mode} gauge has nothing to refit"
            )
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
                rotation, gauge_stats = fit_gauge_rotation(
                    targets[index],
                    student_init,
                    mode=gauge_mode,
                    seed=int(getattr(cfg, "gauge_random_seed", 0)),
                    theta=getattr(cfg, "gauge_theta", None),
                )
                if gauge_mode == "procrustes":
                    # Kept for the optional per-epoch re-estimation of R (alternating
                    # minimisation of the gauge-invariant endpoint discrepancy).
                    self._gauge_state = {
                        "targets_pca": targets.clone(),
                        "index": index,
                        "texts": [texts[i] for i in index.tolist()],
                        "history": [gauge_stats],
                    }
                targets = targets @ rotation
                if renormalize:
                    targets = F.normalize(targets, dim=-1, eps=cfg.eps_norm)
                reference = (
                    ""
                    if gauge_mode == "procrustes"
                    else f" (Procrustes would reach {gauge_stats['cos_procrustes']:+.3f})"
                )
                print(
                    f"Fitted {gauge_mode} gauge rotation on "
                    f"{gauge_stats['samples']} sentences: mean student-target cosine "
                    f"{gauge_stats['cos_before']:+.3f} -> "
                    f"{gauge_stats['cos_after']:+.3f}{reference}"
                )
                if gauge_stats.get("endpoint_reflected"):
                    # Said out loud because it changes what theta = 1 *means*: the
                    # curve's right-hand end is a different Haar draw from the one
                    # the 'random' arm with this seed plots, and no continuous path
                    # to that one exists. See interpolate_rotation.
                    print(
                        "  theta=1 endpoint reflected: the Procrustes gauge and this "
                        f"seed's Haar draw lie in different components of O({student_dim}), "
                        "so the geodesic ends at that draw with its last column negated, "
                        "not at the gauge --gauge_rotation random would use"
                    )
                # PR ~ 1 means the cross-covariance is rank-one and the gauge can
                # only match the two mean vectors, so a null R ablation on this pair
                # is predicted rather than surprising.
                print(
                    "Cross-covariance participation ratio "
                    f"{gauge_stats['participation_ratio']:.2f} of {student_dim} "
                    f"(top singular share {gauge_stats['top_singular_share']:.3f})"
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
                    "projection_type": projection_type,
                    "projection_seed": int(getattr(cfg, "projection_seed", 0)),
                    "pca_center_fit": cfg.pca_center_fit,
                    "pca_subtract_mean": cfg.pca_subtract_mean,
                    "explained_energy": explained,
                    "gauge_align": bool(getattr(cfg, "gauge_align", False)),
                    "gauge_rotation": gauge_mode,
                    "gauge_random_seed": int(getattr(cfg, "gauge_random_seed", 0)),
                    # The matrix itself, so a saved run can be replayed exactly.
                    "gauge_matrix": rotation,
                    "gauge_stats": gauge_stats,
                },
                projection_path,
            )
            print(f"Teacher projection saved: {projection_path}")

        return targets

    def _learned_teacher_targets(
        self, teacher_cls: torch.Tensor, projection_type: str
    ) -> torch.Tensor:
        """Targets for the learned-projector baselines: the teacher, left alone.

        Nothing is fitted here. The map is a parameter trained with the student, so
        it cannot be applied once up front the way ``P_T`` is; the cache stays in the
        teacher's own space and the criterion's projector maps it (or the student)
        at every step. That difference *is* the ablation: same targets, same
        objective, and the only question is whether the map may adapt to them.
        """
        cfg = self.config
        direction = LEARNED_PROJECTIONS[projection_type]
        student_dim = self.model_student.config.hidden_size
        teacher_dim = teacher_cls.shape[-1]
        # Read by the criterion constructor, which is where the parameters have to be
        # created: they must exist before the optimizer's param group is added.
        self._learned_projector = {
            "teacher_dim": teacher_dim,
            "student_dim": student_dim,
            "direction": direction,
        }
        targets = F.normalize(
            teacher_cls.detach().to(torch.float32), p=2, dim=-1, eps=cfg.eps_norm
        )
        mapping = (
            f"{teacher_dim} -> {student_dim}"
            if direction == "t2s"
            else f"{student_dim} -> {teacher_dim} (student mapped up)"
        )
        print(
            f"Learned target map {projection_type} ({mapping}): no map is fitted, "
            "targets stay in the teacher space and the projection is trained with "
            "the student"
        )
        if getattr(cfg, "gauge_align", False):
            # Not an ignored flag but an inapplicable one: a gauge fixes the
            # arbitrary orientation of a *frozen* basis, and a learned map has no
            # fixed basis to orient. Said out loud because gauge_align is on by
            # default, so a learned run would otherwise look gauge-aligned in the log.
            print(
                "Gauge alignment does not apply to a learned target map (there is no "
                "frozen basis to orient); gauge_align is ignored for this run"
            )

        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            projection_path = os.path.join(cfg.save_dir, "teacher_projection.pt")
            torch.save(
                {
                    "projection": None,
                    "mean": None,
                    "teacher_model_name": cfg.teacher_model_name,
                    "student_dim": student_dim,
                    "teacher_dim": teacher_dim,
                    "projection_type": projection_type,
                    "learned_direction": direction,
                    # A learned map keeps everything only in the sense that nothing
                    # is discarded before training; what it keeps is what it learns.
                    "explained_energy": None,
                    "gauge_align": False,
                    "gauge_rotation": "none",
                    "gauge_matrix": None,
                    "gauge_stats": None,
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
        the alternation descends min_{theta, R} L_end(Z_theta, T R) monotonically.
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
                saved["gauge_matrix"] = rotation
                saved["gauge_history"] = state["history"]
                torch.save(saved, path)

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
            param_groups = [{"params": optimizer_parameters, "lr": cfg.learning_rate}]

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

    def _pool_student(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Sentence vector of the student, pooled the way the benchmarks read it."""
        if getattr(self.config, "student_pooling", "cls") == "mean":
            return mean_pooling(last_hidden_state, attention_mask)
        return last_hidden_state[:, 0, :]

    def _pooled_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """One student forward, returned as a pooled sentence vector."""
        out = self.model_student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return self._pool_student(out.last_hidden_state, attention_mask)

    def _dropout_pair_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        hidden_states: bool,
    ):
        """Both dropout views of one batch, in a single forward over ``2B`` rows.

        A "dropout view" is the same sentence encoded again: the two passes differ
        only by the dropout masks they draw. Stacking the batch on itself draws an
        independent mask per row exactly as two passes do -- dropout is sampled
        per element, not per call -- so the pair is the same pair, at one kernel
        launch per operation instead of two and at a batch size that keeps a small
        student's matmuls out of the launch-bound regime.

        What it is not is bit-identical to two passes: one RNG draw of ``2B`` masks
        consumes the generator differently from two draws of ``B``. The trajectory
        of a seeded run therefore changes, which is why ``--no-fused_views`` exists
        for reproducing numbers collected before it.

        Returns ``(output, second_view)`` where ``output`` holds the *first* view's
        states, sliced back to ``B`` rows, and ``second_view`` is the second view's
        pooled (unnormalised) final state.
        """
        rows = input_ids.shape[0]
        out = self.model_student(
            input_ids=torch.cat([input_ids, input_ids], dim=0),
            attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
            output_hidden_states=hidden_states,
            return_dict=True,
        )
        first = SimpleNamespace(
            last_hidden_state=out.last_hidden_state[:rows],
            hidden_states=(
                tuple(state[:rows] for state in out.hidden_states)
                if hidden_states
                else None
            ),
        )
        second_view = self._pool_student(out.last_hidden_state[rows:], attention_mask)
        return first, second_view

    def _fuses_dropout_views(self, view_setting: str) -> bool:
        """Whether the two views of this step are the same input under dropout."""
        return (
            bool(getattr(self.config, "fused_views", True))
            and view_setting == "dropout"
        )

    @staticmethod
    def _view_inputs(
        batch_s: dict[str, torch.Tensor], mode: str, setting: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs of the second view of a contrastive term.

        "dropout" re-encodes the same sentence, so the two views differ only by the
        dropout masks; "pair" takes the row's second sentence as the positive.
        """
        if mode == "dropout":
            return batch_s["input_ids1_stu"], batch_s["attention_mask1_stu"]
        if mode == "pair":
            return batch_s["input_ids2_stu"], batch_s["attention_mask2_stu"]
        raise ValueError(
            f"Unsupported {setting}={mode!r}; expected 'dropout' or 'pair'"
        )

    def _student_batch(
        self, batch: dict, extra: tuple[str, ...] = ()
    ) -> dict[str, torch.Tensor]:
        """The student-side tensors of a batch, on the student device.

        ``extra`` names the cached-teacher tensors a method also reads there
        (``teacher_cls``, ``teacher_topo``): they supervise the student, so they
        travel with the student half of the batch.
        """
        return {
            key: value.to(self.device_s, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
            and (key.endswith("_stu") or key == "labels" or key in extra)
        }

    def _teacher_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """The teacher-side tensors of a batch, on the teacher device."""
        return {
            key: value.to(self.device_t, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value) and key.endswith("_tea")
        }

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Backward, step under the grad scaler, and advance the LR schedule."""
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

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

    # ------------------------------------------------------------------ train step

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One optimizer step of the selected method.

        The cached-teacher methods (talas/geoode/rkd) and the teacher-free control
        (simcse) each have their own step; cdm/dskd/emo/stella share one, because
        they all run the teacher online and differ only in the KD term.
        """
        steps = {
            "talas": self._train_step_talas,
            "geoode": self._train_step_geoode,
            "rkd": self._train_step_rkd,
            "simcse": self._train_step_simcse,
        }
        return steps.get(self.config.distill_method, self._train_step_online)(batch)

    # -- cached-teacher methods ------------------------------------------------

    def _talas_forward(self, batch_s: dict) -> tuple[torch.Tensor, dict]:
        """One TALAS pass: the task loss plus the states its criterion reads.

        SAM takes two forward/backward passes per step over the same batch, and this
        is the pass both of them run.
        """
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
        task_loss, _ = info_nce(
            s_out1.last_hidden_state[:, 0, :],
            s_out2.last_hidden_state[:, 0, :],
            temperature=self.config.temperature,
        )
        return task_loss, {
            "hidden_states": s_out1.hidden_states,
            "last_hidden_state": s_out1.last_hidden_state,
        }

    def _init_talas(self, hidden_states, teacher_cls: torch.Tensor) -> None:
        """Build the TALAS criterion, its SAM optimizer and the schedule.

        Deferred to the first batch because the criterion needs one projection head
        per student layer, and the layer count is read off the first forward pass.
        """
        cfg = self.config
        num_layers = len(hidden_states)
        self.criterion = TeacherAnchorKD(
            student_dim=self.model_student.config.hidden_size,
            teacher_dim=teacher_cls.shape[-1],
            num_layers=num_layers,
            last_layer_idx=cfg.last_layer_idx,
            start_rkd=cfg.start_rkd,
            w_task=cfg.w_task,
            w_kd=cfg.w_kd,
            w_struct=cfg.w_struct,
            eps_norm=cfg.eps_norm,
        ).to(self.device_s)

        if not SAM_AVAILABLE:
            raise RuntimeError(
                "SAM optimizer not available. Install pytorch_optimizer."
            )
        rho = getattr(cfg, "rho", 0.05)
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
            optim.AdamW,
            rho=rho,
            adaptive=True,
        )
        self.scheduler = self._build_scheduler()

        total_steps = len(self.train_loader) * cfg.epochs
        print(
            f"Initialized TeacherAnchorKD: {self.model_student.config.hidden_size} -> "
            f"{teacher_cls.shape[-1]}, num_layers={num_layers}, "
            f"last_layer_idx={cfg.last_layer_idx}, start_rkd={cfg.start_rkd}"
        )
        print(f"Initialized SAM optimizer with rho={rho}")
        print(
            f"Initialized scheduler: {total_steps} steps, "
            f"warmup={int(total_steps * cfg.warmup_ratio)}"
        )

    def _train_step_talas(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch_s = self._student_batch(batch, extra=("teacher_cls",))
        teacher_cls = batch_s["teacher_cls"]

        # ========== FIRST PASS ==========
        with autocast("cuda", enabled=torch.cuda.is_available()):
            task_loss, student_outputs = self._talas_forward(batch_s)
            if self.criterion is None:
                self._init_talas(student_outputs["hidden_states"], teacher_cls)
            loss, metrics = self.criterion(
                student_outputs=student_outputs,
                teacher_cls=teacher_cls,
                task_loss=task_loss,
            )
            loss = loss.float()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if not grads_are_finite(self.optimizer):
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            return loss, {**metrics, "skip": "grad_inf_p1"}

        self.optimizer.first_step(zero_grad=True)

        # ========== SECOND PASS ==========
        with autocast("cuda", enabled=torch.cuda.is_available()):
            task_loss_2, student_outputs_2 = self._talas_forward(batch_s)
            loss_2, _ = self.criterion(
                student_outputs=student_outputs_2,
                teacher_cls=teacher_cls,
                task_loss=task_loss_2,
            )
            loss_2 = loss_2.float()

        if not is_finite(loss_2):
            raise RuntimeError(
                f"loss_2 NaN/Inf at epoch={self.current_epoch} step={self.current_step}"
            )

        # The second backward is deliberately unscaled: the parameters are already
        # at the SAM-perturbed point and the scaler has spent its scale on pass 1.
        loss_2.backward()

        if not grads_are_finite(self.optimizer):
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            return loss, {**metrics, "skip": "grad_inf_p2"}

        self.optimizer.second_step(zero_grad=True)
        self.scaler.update()
        self.scheduler.step()

        return loss, metrics

    def _train_step_geoode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        batch_s = self._student_batch(
            batch, extra=("teacher_cls", "teacher_topo", "teacher_deaths")
        )
        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            attention_mask = batch_s["attention_mask1_stu"]

            # Eq. (37) needs a second view of the *same* sentence. The default
            # runs the encoder twice so the two views differ only by dropout;
            # "pair" instead reuses the paired sentence already in the batch.
            second_view = None
            if cfg.lambda_ctr > 0 and self._fuses_dropout_views(cfg.contrastive_view):
                s_out, second_view = self._dropout_pair_forward(
                    batch_s["input_ids1_stu"], attention_mask, hidden_states=True
                )
            else:
                s_out = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if cfg.lambda_ctr > 0:
                    view_ids, view_mask = self._view_inputs(
                        batch_s, cfg.contrastive_view, "contrastive_view"
                    )
                    second_view = self._pooled_forward(view_ids, view_mask)

            loss, metrics = self.criterion(
                hidden_states=s_out.hidden_states,
                teacher=batch_s["teacher_cls"],
                attention_mask=attention_mask,
                second_view=second_view,
                teacher_topo=batch_s.get("teacher_topo"),
                teacher_deaths=batch_s.get("teacher_deaths"),
            )
            loss = loss.float()

        self._optimizer_step(loss)
        return loss, metrics

    def _train_step_rkd(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        batch_s = self._student_batch(batch, extra=("teacher_cls",))
        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            student_cls = self._pooled_forward(
                batch_s["input_ids1_stu"], batch_s["attention_mask1_stu"]
            )
            # The same in-batch contrastive term every other cached-teacher
            # method carries, so this row differs from them by its KD term only.
            second_cls = self._pooled_forward(
                batch_s["input_ids2_stu"], batch_s["attention_mask2_stu"]
            )
            task_loss, _ = info_nce(
                student_cls, second_cls, temperature=cfg.temperature
            )

            loss, metrics = self.criterion(
                student_cls, batch_s["teacher_cls"], task_loss
            )
            loss = loss.float()

        self._optimizer_step(loss)
        return loss, metrics

    def _train_step_simcse(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        batch_s = self._student_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            if self._fuses_dropout_views(cfg.simcse_view):
                out, view2 = self._dropout_pair_forward(
                    batch_s["input_ids1_stu"],
                    batch_s["attention_mask1_stu"],
                    hidden_states=False,
                )
                view1 = self._pool_student(
                    out.last_hidden_state, batch_s["attention_mask1_stu"]
                )
            else:
                view_ids, view_mask = self._view_inputs(
                    batch_s, cfg.simcse_view, "simcse_view"
                )
                view1 = self._pooled_forward(
                    batch_s["input_ids1_stu"], batch_s["attention_mask1_stu"]
                )
                view2 = self._pooled_forward(view_ids, view_mask)
            loss, metrics = self.criterion(view1, view2)
            loss = loss.float()

        self._optimizer_step(loss)
        return loss, metrics

    # -- online-teacher methods (cdm, dskd, emo, stella) ------------------------

    @torch.no_grad()
    def _encode_teacher(self, batch_t: dict, need_atts: bool) -> SimpleNamespace:
        """Run the frozen teacher and move its outputs onto the student device.

        ``no_grad``, not ``inference_mode``: these tensors are consumed by losses
        that save them for backward, and inference tensors cannot be.
        """

        def to_student(tensor):
            return tensor.to(self.device_s, non_blocking=True)

        t_out1 = self.model_teacher(
            input_ids=batch_t["input_ids1_tea"],
            attention_mask=batch_t["attention_mask1_tea"],
            output_attentions=need_atts,
            return_dict=True,
        )
        encoded = SimpleNamespace(
            last1=to_student(t_out1.last_hidden_state),
            cls1=to_student(
                self._pool_teacher(
                    t_out1.last_hidden_state, batch_t["attention_mask1_tea"]
                )
            ),
            atts1=None,
            last2=None,
            atts2=None,
        )

        if need_atts:
            encoded.atts1 = tuple(to_student(att) for att in t_out1.attentions)
            if "input_ids2_tea" in batch_t:
                t_out2 = self.model_teacher(
                    input_ids=batch_t["input_ids2_tea"],
                    attention_mask=batch_t["attention_mask2_tea"],
                    output_attentions=True,
                    return_dict=True,
                )
                encoded.last2 = to_student(t_out2.last_hidden_state)
                encoded.atts2 = tuple(to_student(att) for att in t_out2.attentions)
        return encoded

    def _encode_student(self, batch_s: dict, method: str) -> SimpleNamespace:
        """Run the student over both sides of the batch.

        The three student shapes differ only in their forward signature: StellaModel
        takes neither ``output_attentions`` nor ``return_dict`` and returns its heads
        in a dict, EMO needs the attention maps, and cdm/dskd take a plain forward.
        """
        if method == "stella":
            out1 = self.model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
            )
            out2 = self.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
            )
            return SimpleNamespace(
                out1=out1,
                out2=out2,
                last1=None,
                last2=None,
                cls1=out1["pooled"],
                cls2=out2["pooled"],
            )

        extra = {"output_attentions": True} if method == "emo" else {}
        out1 = self.model_student(
            input_ids=batch_s["input_ids1_stu"],
            attention_mask=batch_s["attention_mask1_stu"],
            return_dict=True,
            **extra,
        )
        out2 = None
        if method != "emo" or "input_ids2_stu" in batch_s:
            out2 = self.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
                return_dict=True,
                **extra,
            )
        last2 = None if out2 is None else out2.last_hidden_state
        return SimpleNamespace(
            out1=out1,
            out2=out2,
            last1=out1.last_hidden_state,
            last2=last2,
            cls1=out1.last_hidden_state[:, 0, :],
            cls2=None if last2 is None else last2[:, 0, :],
        )

    def _train_step_online(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        method = cfg.distill_method
        batch_s = self._student_batch(batch)
        batch_t = self._teacher_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            teacher = self._encode_teacher(batch_t, need_atts=method == "emo")
            student = self._encode_student(batch_s, method)

            # EMO may train a supervised head; cdm/dskd share the in-batch
            # contrastive term; both Stella stages carry their own task term
            # inside the stage loss, so none is computed for it here.
            loss_task, task_metrics = None, {}
            if method == "emo":
                loss_task, task_metrics = self._compute_task_loss(
                    student.cls1, student.cls2, batch_s
                )
            elif method != "stella":
                loss_task, _ = info_nce(
                    student.cls1, student.cls2, temperature=cfg.temperature
                )

            if method == "cdm":
                loss, metrics = self._cdm_loss(
                    batch, batch_s, batch_t, student, teacher, loss_task
                )
            elif method == "dskd":
                loss, metrics = self._dskd_loss(
                    batch_s, batch_t, student, teacher, loss_task
                )
            elif method == "emo":
                loss, metrics = self._emo_loss(
                    batch_s, batch_t, student, teacher, loss_task, task_metrics
                )
            elif method == "stella":
                loss, metrics = self._stella_loss(student, teacher)
            else:
                raise ValueError(f"Unknown distillation method: {method}")

            loss = loss.float()

        self._optimizer_step(loss)
        return loss, metrics

    def _cdm_loss(self, batch, batch_s, batch_t, student, teacher, loss_task):
        cfg = self.config
        keep_s1 = batch_s["attention_mask1_stu"].bool() & (
            ~batch_s["special_tokens_mask1_stu"].bool()
        )
        keep_t1 = batch_t["attention_mask1_tea"].to(self.device_s).bool() & (
            ~batch_t["special_tokens_mask1_tea"].to(self.device_s).bool()
        )

        kd_dtw = self.criterion.compute_cdm_loss(
            S_last=student.last1,
            T_last=teacher.last1,
            batch_input_ids_stu=batch["input_ids1_stu"],
            batch_input_ids_tea=batch["input_ids1_tea"],
            keep_mask_stu=keep_s1,
            keep_mask_tea=keep_t1,
            proj_s2t=self.proj_s2t,
            device_s=self.device_s,
            epoch=self.current_epoch,
            step=self.current_step,
        )

        kd_cls = F.mse_loss(
            F.normalize(self.proj_s2t(student.cls1), p=2, dim=-1),
            F.normalize(teacher.cls1, p=2, dim=-1),
        )

        loss = (
            cfg.w_task * loss_task + cfg.alpha_dtw * kd_dtw * 100 + cfg.w_cls * kd_cls
        )
        metrics = {
            "loss_total": loss.item(),
            "loss_task": loss_task.item(),
            "loss_kd_dtw": kd_dtw.item()
            if isinstance(kd_dtw, torch.Tensor)
            else kd_dtw,
            "loss_kd_cls": kd_cls.item(),
        }
        return loss, metrics

    def _dskd_loss(self, batch_s, batch_t, student, teacher, loss_task):
        special_teacher = batch_t.get("special_tokens_mask1_tea")
        return self.criterion.compute_dskd_loss(
            S_last=student.last1,
            T_last=teacher.last1,
            S_cls=student.cls1,
            T_cls=teacher.cls1,
            mask_student=batch_s["attention_mask1_stu"],
            mask_teacher=batch_t["attention_mask1_tea"].to(self.device_s),
            task_loss=loss_task,
            special_tokens_mask_student=batch_s.get("special_tokens_mask1_stu"),
            special_tokens_mask_teacher=None
            if special_teacher is None
            else special_teacher.to(self.device_s),
            device=self.device_s,
        )

    def _emo_loss(self, batch_s, batch_t, student, teacher, loss_task, task_metrics):
        cfg = self.config
        att_loss_weight = getattr(cfg, "att_loss_weight", 0.1)
        ot_loss_weight = getattr(cfg, "ot_loss_weight", 1.0)

        def side(index: int, student_last, student_atts, teacher_last, teacher_atts):
            return self.criterion.compute_emo_loss(
                teacher_outputs=SimpleNamespace(
                    last_hidden_state=teacher_last, attentions=teacher_atts
                ),
                student_outputs=SimpleNamespace(
                    last_hidden_state=student_last, attentions=student_atts
                ),
                input_ids_tea=batch_t[f"input_ids{index}_tea"].to(self.device_s),
                input_ids_stu=batch_s[f"input_ids{index}_stu"],
                attention_mask_tea=batch_t[f"attention_mask{index}_tea"].to(
                    self.device_s
                ),
                attention_mask_stu=batch_s[f"attention_mask{index}_stu"],
                tok_teacher=self.tok_teacher,
                tok_student=self.tok_student,
                att_loss_weight=att_loss_weight,
                ot_loss_weight=ot_loss_weight,
            )

        kd_loss, kd_metrics = side(
            1, student.last1, student.out1.attentions, teacher.last1, teacher.atts1
        )
        if student.last2 is not None and teacher.last2 is not None:
            kd_loss2, kd_metrics2 = side(
                2, student.last2, student.out2.attentions, teacher.last2, teacher.atts2
            )
            kd_loss = 0.5 * (kd_loss + kd_loss2)
            kd_metrics = {
                key: 0.5 * (kd_metrics[key] + kd_metrics2[key]) for key in kd_metrics
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
        return loss, metrics

    def _stella_loss(self, student, teacher):
        cfg = self.config
        if self.current_stage == 1:
            return stella_stage1_loss(
                student.out1["fc1"],
                teacher.cls1,
                w_cos=getattr(cfg, "w_cos_stage1", 10.0),
                w_sim=getattr(cfg, "w_sim_stage1", 200.0),
                w_tri=getattr(cfg, "w_tri_stage1", 20.0),
            )
        return stella_stage2_loss(
            student.cls1,
            student.cls2,
            student.out1["fc1"],
            student.out1["fc2"],
            student.out1["fc3"],
            student.out1["fc4"],
            teacher.cls1,
            temperature=cfg.temperature,
            w_task=cfg.w_task,
            w_cos=getattr(cfg, "w_cos_stage2", 10.0),
            w_sim=getattr(cfg, "w_sim_stage2", 200.0),
            w_tri=getattr(cfg, "w_tri_stage2", 20.0),
        )

    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch

        total_loss = 0.0
        n_items = 0
        metric_totals = {}
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

        timer = StepTimer()
        timed = 0  # steps whose duration has been read back so far

        for step, batch in enumerate(pbar):
            self.current_step = step

            timer.start()
            loss, metrics = self.train_step(batch)
            timer.stop()

            # Whatever the GPU has finished by now, at no cost; the rest is drained
            # once at the end of the epoch.
            for seconds in timer.take_new():
                if timed >= self.warmup_steps:
                    self.step_times.append(seconds)
                    self.ma_window.append(seconds)
                timed += 1
            dt = self.ma_window[-1] if self.ma_window else 0.0
            self.global_step += 1
            if getattr(self, "use_wandb", False) and WANDB_AVAILABLE:
                log_payload = {
                    "train/epoch": epoch + 1,
                    "train/global_step": self.global_step,
                    # The most recent step whose CUDA events have completed, which
                    # trails the current one by however deep the queue is. It is a
                    # real step duration either way; the exactly-aligned series is
                    # in the per-step records written at the end of the epoch.
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
                # Filled in below, once the epoch's event timings have been drained.
                "step_seconds": 0.0,
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

            if step >= self.warmup_steps and self.step_times:
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

        # The one synchronisation of the epoch: read back every step's duration and
        # put it on the record that was buffered for it.
        epoch_step_times = timer.finish()
        for record, seconds in zip(step_records, epoch_step_times):
            record["step_seconds"] = float(seconds)

        if len(self.step_times) > 0:
            epoch_avg = sum(self.step_times) / len(self.step_times)
            print(
                f"[Epoch {epoch + 1}] Avg step time = {epoch_avg * 1000:.2f} ms "
                f"({1.0 / epoch_avg:.2f} it/s)"
            )

        print(f"Done train_epoch {epoch + 1}")
        self.log_step_records(step_records)
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
            "ndcg_at_10": "nDCG@10",
            "recall_at_10": "R@10",
            "mrr_at_10": "MRR@10",
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
        """Average the primary scores over IOD, OOD, retrieval and all benchmarks.

        Every benchmark contributes its own primary metric (macro-F1, AP, Spearman
        or nDCG@10), all on a 0-1 scale, so the groups are unweighted means over
        benchmarks rather than over examples.

        The IOD/OOD split covers the sentence-level probes only: retrieval is held
        out from both and reported on its own row, so AVG (IOD) and AVG (OOD) keep
        meaning what they meant before retrieval was added. AVG (ALL) spans
        everything.
        """
        sentence_level = [
            name for name in scores_by_benchmark if name not in RETRIEVAL_BENCHMARKS
        ]
        groups = {
            "avg_iod": (
                "AVG (IOD)",
                sorted(name for name in sentence_level if name in IOD_BENCHMARKS),
            ),
            "avg_ood": (
                "AVG (OOD)",
                sorted(name for name in sentence_level if name not in IOD_BENCHMARKS),
            ),
            "avg_retrieval": (
                "AVG (RETRIEVAL)",
                sorted(
                    name for name in scores_by_benchmark if name in RETRIEVAL_BENCHMARKS
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
    ) -> dict[str, float]:
        primary_metrics = {
            "classification": "f1",
            "pair": "average_precision",
            "sts": "spearman",
            "retrieval": "ndcg_at_10",
        }
        rows = []
        scores_by_benchmark: dict[str, float] = {}
        for family in ("classification", "pair", "sts", "retrieval"):
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

    def _ensure_pair_thresholds(self) -> None:
        """Run one validation pass if the final test needs thresholds it never got.

        With eval_every=0 no per-epoch validation runs, but the pair benchmarks
        still calibrate their threshold on validation unless pair_threshold_source
        is "test". Doing that pass here keeps the test score held out instead of
        failing the run after training has finished.
        """
        source = getattr(self.config, "pair_threshold_source", "validation")
        if source != "validation":
            return
        if getattr(self, "pair_validation_thresholds", None) is not None:
            return
        print(
            "No validation pass has run (eval_every=0); running one now to select "
            "the pair thresholds for the final test evaluation"
        )
        validation_results = self.evaluate("validation")
        self.log_experiment_record({"validation": validation_results})

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
        # Retrieval has no validation qrels and embedding the three corpora is ~92k
        # forward passes, so it is scored on the test split only.
        retrieval = {}
        if split == "test" and getattr(self.config, "eval_retrieval", True):
            retrieval = eval_retrieval_task(
                student_model, test_retrieval_tasks, self.tok_student
            )
        if split == "validation":
            self.pair_validation_thresholds = selected_thresholds
        results = {
            "classification": classification,
            "pair": pair,
            "sts": sts,
            "retrieval": retrieval,
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

    def _log_eval_to_wandb(self, prefix: str, results: dict | None) -> None:
        if not getattr(self, "use_wandb", False) or not WANDB_AVAILABLE:
            return
        if results is None:
            return
        wandb.log(self._flatten_metrics(prefix, results), step=self.global_step)

    def _run_evaluation(self, split: str) -> dict | None:
        """Evaluate ``split``, logging to W&B. A failed pass warns and returns None.

        Evaluation is a report on the run, not part of it: a benchmark file that is
        missing or malformed must not throw away the epochs already trained.
        """
        try:
            results = self.evaluate(split)
        except Exception as error:
            print(f"Warning: {split} evaluation failed with error: {error}")
            print("Continuing training...")
            return None
        self._log_eval_to_wandb(split, results)
        return results

    def _final_test_evaluation(
        self, reusable: dict | None = None, extra: dict | None = None
    ) -> None:
        """Score the test split once at the end of training and record it.

        ``reusable`` is the last epoch's results when that epoch already scored the
        test split on these same weights; re-running would only burn a second pass.
        """
        try:
            if reusable is not None:
                print("Reusing the final epoch's test evaluation")
                test_results = reusable
                self.print_evaluation_table("test", test_results, final=True)
            else:
                self._ensure_pair_thresholds()
                test_results = self.evaluate("test", final=True)
            self._log_eval_to_wandb("test", test_results)
            self.log_experiment_record({**(extra or {}), "test": test_results})
        except Exception as error:
            print(f"Warning: Final test evaluation failed with error: {error}")

    def train(self):
        if self.config.distill_method == "stella":
            self._train_stella()
        else:
            self._train_single_stage()

    def _train_stella(self):
        cfg = self.config
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
        for head in (
            self.model_student.fc2,
            self.model_student.fc3,
            self.model_student.fc4,
        ):
            for p in head.parameters():
                p.requires_grad = False

        print("Frozen: backbone, fc2, fc3, fc4")
        print("Trainable: fc1")

        self.current_stage = 1
        for epoch in range(cfg.epochs_stage1):
            avg_loss = self.train_epoch(epoch)
            self.log_experiment_record({"stage": 1, "train": self.last_epoch_metrics})

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
        stage2_split = "test" if cfg.evaluate_test_each_epoch else "validation"
        for epoch in range(cfg.epochs_stage2):
            avg_loss = self.train_epoch(epoch)

            print("\n" + "=" * 60)
            print(f"Evaluation after Stage2 Epoch {epoch + 1}")
            print("=" * 60)
            epoch_results = self._run_evaluation(stage2_split)
            print("=" * 60 + "\n")

            self.log_experiment_record(
                {
                    "stage": 2,
                    "train": self.last_epoch_metrics,
                    stage2_split: epoch_results,
                }
            )

            if should_save_epoch(epoch, cfg.save_every):
                self.save_checkpoint(epoch, {"loss": avg_loss})

        print("\n" + "=" * 70)
        print("STAGE 2 COMPLETED!")
        print("=" * 70)

        self.save_checkpoint(cfg.epochs_stage2 - 1, {"loss": avg_loss})
        self._final_test_evaluation(extra={"stage": 2})

        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)

    def _train_single_stage(self):
        cfg = self.config
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        print(f"Method: {cfg.distill_method}")
        print(f"Student: {cfg.student_model_name}")
        if cfg.distill_method == "simcse":
            print(f"Teacher: none (control for {cfg.teacher_model_name})")
        else:
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
                epoch_results = self._run_evaluation(eval_split)

            print("=" * 60 + "\n")
            # Keyed by the split that actually ran, so a reader of metrics.jsonl
            # never has to guess which data a per-epoch number came from.
            self.log_experiment_record(
                {"train": self.last_epoch_metrics, eval_split: epoch_results}
            )

            if should_save_epoch(epoch, cfg.save_every):
                try:
                    self.save_checkpoint(epoch, {"loss": avg_loss})
                except Exception as error:
                    if getattr(cfg, "weights_dir", None):
                        raise RuntimeError(
                            f"Required epoch {epoch + 1} weights could not be saved"
                        ) from error
                    print(f"Warning: Saving checkpoint failed with error: {error}")
                    print("Continuing training...")

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print("Done train()")

        self.save_checkpoint(cfg.epochs - 1, {"loss": avg_loss})
        # The last epoch already scored the test split under this setting, on the
        # same weights, whenever the cadence lands on it.
        reusable = (
            epoch_results
            if eval_split == "test"
            and epoch_results is not None
            and cfg.eval_every
            and cfg.epochs % cfg.eval_every == 0
            else None
        )
        self._final_test_evaluation(reusable)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
